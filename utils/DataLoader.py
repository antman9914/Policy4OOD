from torch.utils.data import Dataset, DataLoader


class CustomizedDataset(Dataset):
    def __init__(self, raw_data, kg_idx, window_size):

        super(CustomizedDataset, self).__init__()
        input_window, output_window = window_size
        self.raw_data = raw_data
        self.data_size = raw_data.size(1) - input_window - output_window + 1
        self.data, self.label = [], []
        self.policy_emb = []
        self.policy_emb_set = []
        self.future_policy = []
        self.future_policy_set = []
        for i in range(self.data_size):
            self.data.append(self.raw_data[:, i:(i+input_window), :])
            self.label.append(self.raw_data[:, (i+input_window):(i+input_window+output_window), 0])
            
            self.policy_emb_set.append(kg_idx[:, i:(i+input_window)])
            self.future_policy_set.append(kg_idx[:, (i+input_window):(i+input_window+output_window)])
            
    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx], self.policy_emb_set[idx], self.future_policy_set[idx]

    def __len__(self):
        return self.data_size


def get_idx_data_loader(raw_data, kg_idx, window_size, batch_size, shuffle):
    dataset = CustomizedDataset(raw_data, kg_idx, window_size)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader
