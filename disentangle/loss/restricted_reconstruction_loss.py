import torch
import torch.nn as nn


class RestrictedReconstruction:

    def __init__(self,
                 w_split,
                 w_recons,
                 finegrained_restriction=True,
                 finegrained_restriction_retain_positively_correlated=True,
                 correct_grad_retain_negatively_correlated=True) -> None:
        self._w_split = w_split
        self._w_recons = w_recons
        self._finegrained_restriction = finegrained_restriction
        self._finegrained_restriction_retain_positively_correlated = finegrained_restriction_retain_positively_correlated
        self._correct_grad_retain_negatively_correlated = correct_grad_retain_negatively_correlated
        print(f'[{self.__class__.__name__}] w_split: {self._w_split}, w_recons: {self._w_recons}')

    @staticmethod
    def get_grad_direction(score, params):
        grad_all = torch.autograd.grad(score, params, create_graph=False, retain_graph=True, allow_unused=True)
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
                           retain_positively_correlated=False,
                           retain_negatively_correlated=False):
        grad_components = []
        assert int(along_direction) + int(orthogonal_direction) + int(retain_positively_correlated) + int(
            retain_negatively_correlated) == 1, 'Donot be lazy. Set one of the booleans to True.'
        assert isinstance(along_direction, bool)
        assert isinstance(orthogonal_direction, bool)
        assert isinstance(retain_positively_correlated, bool)
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
                elif retain_positively_correlated:
                    if component < 0:
                        grad_components.append(grad_vector - grad_direction * component)
                    else:
                        neg_corr_count += 1
                        grad_components.append(grad_vector)
                elif retain_negatively_correlated:
                    if component > 0:
                        grad_components.append(grad_vector - grad_direction * component)
                    else:
                        neg_corr_count += 1
                        grad_components.append(grad_vector)

        # print('Retained neg corr fraction', neg_corr_count / len(grad_vectors))

        # check one grad for norm
        assert torch.norm(grad_direction) - 1 < 1e-6

        return grad_components

    def loss_fn(self, tar, pred):
        return torch.mean((tar - pred)**2)

    def get_correct_grad(self, params, normalized_input, normalized_target, normalized_target_prediction,
                         normalized_input_prediction):
        unsup_reconstruction_loss = self.loss_fn(normalized_input, normalized_input_prediction)
        incorrect_c1loss = self.loss_fn(normalized_target[:, 0], normalized_target_prediction[:, 1])
        incorrect_c2loss = self.loss_fn(normalized_target[:, 1], normalized_target_prediction[:, 0])

        incorrect_c1_all = self.get_grad_direction(incorrect_c1loss, params)
        incorrect_c2_all = self.get_grad_direction(incorrect_c2loss, params)
        if self._finegrained_restriction:
            correct_loss = self.loss_fn(normalized_target, normalized_target_prediction)
            correct_grad_all = self.get_grad_direction(correct_loss, params)
            incorrect_c1_all = self.get_grad_component(
                incorrect_c1_all,
                correct_grad_all,
                retain_negatively_correlated=self._finegrained_restriction_retain_positively_correlated,
                orthogonal_direction=not self._finegrained_restriction_retain_positively_correlated)
            incorrect_c2_all = self.get_grad_component(
                incorrect_c2_all,
                correct_grad_all,
                retain_negatively_correlated=self._finegrained_restriction_retain_positively_correlated,
                orthogonal_direction=not self._finegrained_restriction_retain_positively_correlated)

        unsup_grad_all = torch.autograd.grad(unsup_reconstruction_loss,
                                             params,
                                             create_graph=False,
                                             retain_graph=True,
                                             allow_unused=True)

        incorrect_c2_all = self.get_grad_component(incorrect_c2_all, incorrect_c1_all, orthogonal_direction=True)
        corrected_unsup_grad_all = self.get_grad_component(
            unsup_grad_all,
            incorrect_c1_all,
            orthogonal_direction=not self._correct_grad_retain_negatively_correlated,
            retain_negatively_correlated=self._correct_grad_retain_negatively_correlated)
        corrected_unsup_grad_all = self.get_grad_component(
            corrected_unsup_grad_all,
            incorrect_c2_all,
            orthogonal_direction=not self._correct_grad_retain_negatively_correlated,
            retain_negatively_correlated=self._correct_grad_retain_negatively_correlated)
        return corrected_unsup_grad_all, unsup_reconstruction_loss

    def update_gradients(self, params, normalized_input, normalized_target, normalized_target_prediction,
                         normalized_input_prediction):

        if len(normalized_target) == 0:
            print('No target, hence skipping input reconstruction loss')
            return {'input_reconstruction_loss': torch.tensor(0.0)}

        corrected_unsup_grad_all, input_reconstruction_loss = self.get_correct_grad(params, normalized_input,
                                                                                    normalized_target,
                                                                                    normalized_target_prediction,
                                                                                    normalized_input_prediction)
        # split_grad_all, split_loss = self.get_split_grad(params, normalized_target, normalized_target_prediction)
        for param, corrected_unsup_grad in zip(params, corrected_unsup_grad_all):
            if corrected_unsup_grad is None:
                continue

            param.grad = self._w_split * param.grad + self._w_recons * corrected_unsup_grad

        return {'input_reconstruction_loss': input_reconstruction_loss}
