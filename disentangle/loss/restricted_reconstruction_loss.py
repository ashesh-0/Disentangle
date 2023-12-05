import torch
import torch.nn as nn


class RestrictedReconstruction:

    def __init__(self, w_split, w_recons) -> None:
        self._w_split = w_split
        self._w_recons = w_recons

    @staticmethod
    def get_grad_direction(score, params):
        grad_all = torch.autograd.grad(score, params, create_graph=True, allow_unused=True)
        grad_direction = []
        for grad in grad_all:
            if grad is None:
                grad_direction.append(None)
            else:
                grad_direction.append(grad / torch.norm(grad))
        return grad_direction

    def loss_fn(self, tar, pred):
        return torch.mean((tar - pred)**2)

    def get_correct_grad(self, params, normalized_input, normalized_target, normalized_target_prediction,
                         normalized_input_prediction):
        unsup_reconstruction_loss = self.loss_fn(normalized_input, normalized_input_prediction)
        incorrect_c1loss = self.loss_fn(normalized_target[:, 0], normalized_target_prediction[:, 1])
        incorrect_c2loss = self.loss_fn(normalized_target[:, 1], normalized_target_prediction[:, 0])
        incorrect_c1_all = self.get_grad_direction(incorrect_c1loss, params)
        incorrect_c2_all = self.get_grad_direction(incorrect_c2loss, params)
        unsup_grad_all = torch.autograd.grad(unsup_reconstruction_loss, params, create_graph=True, allow_unused=True)
        corrected_unsup_grad_all = []
        for unsup_grad, incorrect_c1, incorrect_c2 in zip(unsup_grad_all, incorrect_c1_all, incorrect_c2_all):
            if unsup_grad is None:
                corrected_unsup_grad_all.append(None)
                continue
            # making the two directions incorrect_c1 and incorrect_c2 orthogonal. only then the subtraction makes sense.
            incorrect_c2 = incorrect_c2 - torch.dot(incorrect_c2.view(-1, ), incorrect_c1.view(-1, )) * incorrect_c1

            incor_c1_comp = torch.dot(unsup_grad.view(-1, ), incorrect_c1.view(-1, ))
            incor_c2_comp = torch.dot(unsup_grad.view(-1, ), incorrect_c2.view(-1, ))
            unsup_grad_corrected = unsup_grad - incor_c1_comp * incorrect_c1 - incor_c2_comp * incorrect_c2
            # ug = f'{torch.norm(unsup_grad).item():.5f}'
            # ucg = f'{torch.norm(unsup_grad_corrected).item():.5f}'
            # ic1 = f'{incor_c1_comp.item():.5f}'
            # ic2 = f'{incor_c2_comp.item():.5f}'
            # print(f'{torch.norm(unsup_grad).item()/(torch.norm(unsup_grad_corrected).item() + 1e-5)} unsup_grad: {ug}, unsup_grad_corrected: {ucg}, ic1: {ic1}, ic2: {ic2}')
            corrected_unsup_grad_all.append(unsup_grad_corrected)
        return corrected_unsup_grad_all, unsup_reconstruction_loss

    # def get_split_grad(self, params, normalized_target, normalized_target_prediction):
    #     loss = self.loss_fn(normalized_target, normalized_target_prediction)
    #     grad_all = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    #     return grad_all, loss

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
            # import pdb; pdb.set_trace()
            # gn = f'split: {torch.norm(param.grad).item():.5f}'
            # gnc = f'recons: {torch.norm(self._w_recons*corrected_unsup_grad).item():.5f}'
            # ratio = f'{torch.norm(param.grad).item()/(torch.norm(self._w_recons*corrected_unsup_grad).item() + 1e-5):.5f}'
            # print(ratio, gn,gnc)
            # if corrected_unsup_grad.isnan().any():
            #     import pdb;
            #     pdb.set_trace()
            # we assume that split_loss.backward() has been called before.
            param.grad = self._w_split * param.grad + self._w_recons * corrected_unsup_grad

        return {'input_reconstruction_loss': input_reconstruction_loss}
