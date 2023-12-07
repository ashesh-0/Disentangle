import torch
import torch.nn as nn


class RestrictedReconstruction:

    def __init__(self, w_split, w_recons, finegrained_restriction=True) -> None:
        self._w_split = w_split
        self._w_recons = w_recons
        self._finegrained_restriction = finegrained_restriction
        print(f'[{self.__class__.__name__}] w_split: {self._w_split}, w_recons: {self._w_recons}')

    @staticmethod
    def get_grad_direction(score, params, retain_graph=True):
        grad_all = torch.autograd.grad(score, params, retain_graph=retain_graph, allow_unused=True)
        grad_direction = []
        for grad in grad_all:
            if grad is None:
                grad_direction.append(None)
            else:
                grad_direction.append(grad / torch.norm(grad))
        return grad_direction

    @staticmethod
    def get_grad_component(grad_vectors,
                           reference_grad_directions,
                           along_direction=False,
                           orthogonal_direction=False,
                           not_positively_correlated=False):
        grad_components = []
        assert int(along_direction) + int(orthogonal_direction) + int(
            not_positively_correlated
        ) == 1, 'Donot be lazy. Set either along_direction or orthogonal_direction to True.'
        assert isinstance(along_direction, bool)
        assert isinstance(orthogonal_direction, bool)
        assert isinstance(not_positively_correlated, bool)
        neg_corr_count = 0
        for grad_vector, grad_direction in zip(grad_vectors, reference_grad_directions):
            if grad_vector is None:
                grad_components.append(None)
            else:
                component = torch.dot(grad_vector.view(-1), grad_direction.view(-1))
                if along_direction:
                    grad_components.append(grad_direction * component)
                elif orthogonal_direction:
                    grad_components.append(grad_vector - grad_direction * component)
                elif not_positively_correlated:
                    if component > 0:
                        grad_components.append(grad_vector - grad_direction * component)
                    else:
                        neg_corr_count += 1
                        grad_components.append(grad_vector)

        print('Retained neg corr fraction', neg_corr_count / len(grad_vectors))

        # check one grad for norm
        assert torch.norm(grad_direction) - 1 < 1e-6

        return grad_components

    def loss_fn(self, tar, pred):
        return torch.mean((tar - pred)**2)

    def get_incorrect_grad(self, params, normalized_target, normalized_target_prediction, retain_graph=True):
        incorrect_c1loss = self.loss_fn(normalized_target[:, 0], normalized_target_prediction[:, 1])
        incorrect_c2loss = self.loss_fn(normalized_target[:, 1], normalized_target_prediction[:, 0])

        incorrect_c1_all = self.get_grad_direction(incorrect_c1loss, params)
        # if self._finegrained_restriction, then retain graph.
        incorrect_c2_all = self.get_grad_direction(incorrect_c2loss,
                                                   params,
                                                   retain_graph=retain_graph or self._finegrained_restriction)
        if self._finegrained_restriction:
            correct_loss = self.loss_fn(normalized_target, normalized_target_prediction)
            correct_grad_all = self.get_grad_direction(correct_loss, params, retain_graph=retain_graph)
            incorrect_c1_all = self.get_grad_component(incorrect_c1_all,
                                                       correct_grad_all,
                                                       not_positively_correlated=True)
            incorrect_c2_all = self.get_grad_component(incorrect_c2_all,
                                                       correct_grad_all,
                                                       not_positively_correlated=True)

        incorrect_gradients = []
        for incorrect_c1, incorrect_c2 in zip(incorrect_c1_all, incorrect_c2_all):
            if incorrect_c1 is None:
                incorrect_gradients.append((None, None))
                continue
            # making the two directions incorrect_c1 and incorrect_c2 orthogonal. only then the subtraction makes sense.
            incorrect_c2 = incorrect_c2 - torch.dot(incorrect_c2.view(-1, ), incorrect_c1.view(-1, )) * incorrect_c1
            incorrect_gradients.append((incorrect_c1, incorrect_c2))
        return incorrect_gradients

    def get_correct_grad(self, params, normalized_input, normalized_input_prediction, incorrect_gradients):
        unsup_reconstruction_loss = self.loss_fn(normalized_input, normalized_input_prediction)
        unsup_gradients = torch.autograd.grad(unsup_reconstruction_loss, params, retain_graph=False, allow_unused=True)
        output_grads = []
        for unsup_grad, incor_grads in zip(unsup_gradients, incorrect_gradients):
            if unsup_grad is None:
                output_grads.append(None)
                continue

            assert isinstance(incor_grads, tuple) or isinstance(incor_grads, list)
            cor_grad = unsup_grad
            for incor_grad in incor_grads:
                component = torch.dot(cor_grad.view(-1, ), incor_grad.view(-1, ))
                if component > 0:
                    cor_grad -= component * incor_grad

            output_grads.append(cor_grad)
        return output_grads, unsup_reconstruction_loss

    def update_gradients(self,
                         params,
                         normalized_input,
                         normalized_target,
                         normalized_target_prediction,
                         normalized_input_prediction,
                         incorrect_gradients=None):

        if len(normalized_target) == 0 and incorrect_gradients is None:
            print('No target, hence skipping input reconstruction loss')
            return {'input_reconstruction_loss': torch.tensor(0.0)}

        if incorrect_gradients is None:
            incorrect_gradients = self.get_incorrect_grad(params, normalized_target, normalized_target_prediction)

        corrected_unsup_grad_all, input_reconstruction_loss = self.get_correct_grad(params, normalized_input,
                                                                                    normalized_input_prediction,
                                                                                    incorrect_gradients)

        # split_grad_all, split_loss = self.get_split_grad(params, normalized_target, normalized_target_prediction)
        for param, corrected_unsup_grad in zip(params, corrected_unsup_grad_all):
            if corrected_unsup_grad is None:
                continue
            # we assume that split_loss.backward() has been called before.
            param.grad = self._w_split * param.grad + self._w_recons * corrected_unsup_grad

        return {'input_reconstruction_loss': input_reconstruction_loss}


if __name__ == '__main__':
    direction_tensor = [torch.tensor([3.0, 4]), torch.tensor([1.0, 0, 0]), torch.tensor([12, 5.0])]
    direction_tensor = [x / torch.norm(x) for x in direction_tensor]
    grad_tensor = [torch.tensor([1, 2.0]), torch.tensor([5, 6, 1.0]), torch.tensor([1, 1.0])]
    a = RestrictedReconstruction.get_grad_component(grad_tensor, direction_tensor, orthogonal_direction=True)
    b = RestrictedReconstruction.get_grad_component(grad_tensor, direction_tensor, along_direction=True)
    summed = [x + y for x, y in zip(a, b)]
    dot = [torch.dot(x.view(-1), y.view(-1)) for x, y in zip(a, b)]
    dot_along_direction = [torch.dot(x.view(-1), y.view(-1)) for x, y in zip(a, direction_tensor)]
    print('dot along direction. should be 0', dot_along_direction)
    print(summed)
    print(grad_tensor)
    print(dot)
