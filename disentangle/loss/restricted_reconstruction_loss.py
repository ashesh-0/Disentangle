import numpy as np
import torch
import torch.nn as nn

from torchmetrics.regression import PearsonCorrCoef


def sample_from_gmm(count, mean=0.3, std_dev=0.1):
    # Set the parameters of the GMM
    mean1, mean2 = mean, -1 * mean

    # np.random.seed(42)

    def sample_from_pos():
        return np.random.normal(mean1, std_dev, 1)[0]

    def sample_from_neg():
        return np.random.normal(mean2, std_dev, 1)[0]

    samples = []
    for i in range(count):
        if np.random.rand() < 0.5:
            samples.append(sample_from_pos())
        else:
            samples.append(sample_from_neg())

    return samples


class RestrictedReconstruction:

    def __init__(self,
                 w_split,
                 w_recons,
                 finegrained_restriction=True,
                 finegrained_restriction_retain_positively_correlated=False,
                 correct_grad_retain_negatively_correlated=True,
                 randomize_alpha=True,
                 randomize_numcount=8) -> None:
        self._w_split = w_split
        self._w_recons = w_recons
        self._finegrained_restriction = finegrained_restriction
        self._finegrained_restriction_retain_positively_correlated = finegrained_restriction_retain_positively_correlated
        self._correct_grad_retain_negatively_correlated = correct_grad_retain_negatively_correlated
        self._incorrect_samech_alphas = None  #[0.5, 0.8, 0.8, 0.5]
        self._incorrect_othrch_alphas = None  #[0.5, 0.2, -0.2 - 0.5]
        self._randomize_alpha = randomize_alpha
        self._randomize_numcount = randomize_numcount

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
        # assert orthogonal_direction == True, 'For now, only orthogonal direction is supported.'
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

    def get_incorrect_loss_v2(self, normalized_target, normalized_target_prediction):
        assert self._randomize_alpha == True
        ch1_alphas = sample_from_gmm(self._randomize_numcount, mean=0.25)
        ch2_alphas = sample_from_gmm(self._randomize_numcount, mean=0.25)
        incorrect_c1loss = 0
        incorrect_c2loss = 0
        # import pdb; pdb.set_trace()
        for ch1_alpha, ch2_alpha in zip(ch1_alphas, ch2_alphas):
            tar1 = normalized_target[:, 0, :, :] * (1 - ch1_alpha) + normalized_target[:, 1, :, :] * ch2_alpha
            tar2 = normalized_target[:, 1, :, :] * ch1_alpha + normalized_target[:, 0, :, :] * (1 - ch2_alpha)
            incorrect_c1loss += self.loss_fn(tar1, normalized_target_prediction[:, 0, :, :])
            incorrect_c2loss += self.loss_fn(tar2, normalized_target_prediction[:, 1, :, :])
        incorrect_c1loss /= self._randomize_numcount
        incorrect_c2loss /= self._randomize_numcount
        return incorrect_c1loss, incorrect_c2loss

    def get_incorrect_loss(self, normalized_target, normalized_target_prediction):
        othrch_alphas = [1]
        samech_alphas = [0]
        if self._incorrect_othrch_alphas is not None:
            othrch_alphas = self._incorrect_othrch_alphas
            samech_alphas = self._incorrect_samech_alphas
        elif self._randomize_alpha:
            othrch_alphas = sample_from_gmm(self._randomize_numcount)
            # othrch_alphas = [
            #     torch.Tensor(sample_from_gmm(len(normalized_target))).view(-1, 1, 1).type(normalized_input.dtype).to(
            #         normalized_input.device) for _ in range(self._randomize_numcount)
            # ]
            samech_alphas = [1] * self._randomize_numcount

        incorrect_c1loss = 0
        for alpha1, alpha2 in zip(othrch_alphas, samech_alphas):
            tar = normalized_target[:, 0] * alpha1 + normalized_target[:, 1] * alpha2
            incorrect_c1loss += self.loss_fn(tar, normalized_target_prediction[:, 1])
        incorrect_c1loss /= len(samech_alphas)

        incorrect_c2loss = 0
        for alpha1, alpha2 in zip(samech_alphas, othrch_alphas):
            tar = normalized_target[:, 0] * alpha1 + normalized_target[:, 1] * alpha2
            incorrect_c2loss += self.loss_fn(tar, normalized_target_prediction[:, 0])
        incorrect_c2loss /= len(samech_alphas)
        return incorrect_c1loss, incorrect_c2loss

    def get_correct_grad(self, params, normalized_input, normalized_target, normalized_target_prediction,
                         normalized_input_prediction):
        # tar = normalized_target.detach().cpu().numpy()
        # pred = normalized_target_prediction.detach().cpu().numpy()
        # import numpy as np
        # tar1 = tar[:, 0].reshape(-1,)
        # tar2 = tar[:, 1].reshape(-1,)
        # pred1 = pred[:, 0].reshape(-1,)
        # pred2 = pred[:, 1].reshape(-1,)
        # c0 = np.round(np.corrcoef(tar1, tar2), 2)[0,1]
        # c1 = np.round(np.corrcoef(tar1, pred2), 2)[0,1]
        # c2 = np.round(np.corrcoef(tar2, pred1), 2)[0,1]
        # c1_res = np.round(np.corrcoef(tar1, (pred2 - tar2)) , 2)[0,1]
        # c2_res = np.round(np.corrcoef(tar2, (pred1 - tar1)), 2)[0,1]
        # print(f'c0: {c0} c1: {c1}, c2: {c2}, c1_res: {c1_res}, c2_res: {c2_res}')

        # incorrect_c2loss = self.loss_fn(normalized_target[:, 1], normalized_target_prediction[:, 0])
        incorrect_c1loss, incorrect_c2loss = self.get_incorrect_loss_v2(normalized_target, normalized_target_prediction)
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

        unsup_reconstruction_loss = self.loss_fn(normalized_input, normalized_input_prediction)
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
