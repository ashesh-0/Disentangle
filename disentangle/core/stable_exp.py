import torch


class StableExponential:
    def posneg_separation(self, tensor):
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {'filter': [pos, neg], 'value': [pos_tensor, neg_tensor]}

    def pow(self, tensor):
        posneg_dic = self.posneg_separation(tensor)
        pos, neg = posneg_dic['filter']
        pos_tensor, neg_tensor = posneg_dic['value']
        return torch.exp(neg_tensor) * neg + (1 + pos_tensor) * pos

    def log(self, tensor):
        pass
