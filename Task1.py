# %%
import numpy as np
import csv
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

with open('RawData.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    raw_data = []
    for row in csv_reader:
        raw_data.append(row)

attribute_names = raw_data[0]
attribute_names_to_idx = {n: i for i, n in enumerate(attribute_names)}
raw_data = raw_data[1:]

teams = []
for row in raw_data:
    if row[attribute_names_to_idx['Team']] not in teams:
        teams.append(row[2])
teams.sort()
teams_to_idx = {t: i for i, t in enumerate(teams)}


# %%
def group_data_by_games(data):
    games = {}
    game_count = 0
    prev_game = None
    for row in data:
        game = {'Date': row[attribute_names_to_idx['Date']],
                'Season': row[attribute_names_to_idx['Season']],
                'Round': row[attribute_names_to_idx['Round']],
                'Home Team': row[attribute_names_to_idx['Home Team']],
                'Away Team': row[attribute_names_to_idx['Away Team']],
                'Home Score': row[attribute_names_to_idx['Home Score']],
                'Away Score': row[attribute_names_to_idx['Away Score']],
                'Margin': row[attribute_names_to_idx['Margin']]}
        if prev_game is None or game != prev_game:
            if game != prev_game:
                game_count += 1
            games[game_count] = game.copy()
            games[game_count]['Stats'] = [row[10:-2] + [row[-1]]]
            games[game_count]['Brownlow Votes'] = [row[attribute_names_to_idx['Brownlow Votes']]]
            games[game_count]['Players'] = [row[attribute_names_to_idx['Name']]]
            games[game_count]['Teams'] = [row[attribute_names_to_idx['Team']]]
        else:
            games[game_count]['Stats'].append(row[10:-2] + [row[-1]])
            games[game_count]['Brownlow Votes'].append(row[attribute_names_to_idx['Brownlow Votes']])
            games[game_count]['Players'].append(row[attribute_names_to_idx['Name']])
            games[game_count]['Teams'].append(row[attribute_names_to_idx['Team']])
        prev_game = game
    return games


data_by_games = group_data_by_games(raw_data)


# %%
def clean_data(data):
    '''
    remove the noises and outliers, mainly including those matches of final series
    remove the noisy attributes
    :param data: a dict, output of data_by_games()
    :return:
    '''
    new_data = data.copy()
    for i, d in list(new_data.items()):
        round = d['Round']
        if len(d['Players']) != 44 or round in ['GF', 'EF', 'QF', 'PF', 'SF']:
            new_data.pop(i)
    return new_data


cleaned_data = clean_data(data_by_games)


# %%
def transform_strings_to_numbers(data):
    new_data = data.copy()
    for i, d in new_data.items():
        new_data[i]['Stats'] = np.array([list(map(float, dd)) for dd in d['Stats']])
        new_data[i]['Brownlow Votes'] = np.array(list(map(float, d['Brownlow Votes'])), dtype=np.int)
    return new_data


def split_data_for_training_test(data: dict):
    train_data = {}
    test_data = {}
    val_data = {}
    train_count = 0
    test_count = 0
    for i, d in data.items():
        if float(d['Season']) <= 2015:
            train_data[train_count] = d
            train_count += 1
        else:
            test_data[test_count] = d
            test_count += 1

    val_count = 0
    for i in np.random.permutation(range(train_count))[:train_count // 5]:
        val_data[val_count] = train_data[i]
        val_count += 1
    return train_data, val_data, test_data


transformed_data = transform_strings_to_numbers(cleaned_data)
train_data, val_data, test_data = split_data_for_training_test(transformed_data)


def normalize_data(stats):
    means = np.mean(stats, 0)
    std = np.std(stats, 0)
    return (stats-means)/std


def data_augmentation(stats, votes):
    # shuffle the order of players
    idxes = range(stats.shape[0])
    idxes = np.random.permutation(idxes)
    return stats[idxes], votes[idxes]


def add_team_info(stats, teams):
    tm = np.array([teams_to_idx[t] for t in teams])
    return np.column_stack([stats, tm])


def add_win_info(stats, home_team, teams, margin):
    wins = np.array([float(margin) if home_team == t else -float(margin) for t in teams])
    return np.column_stack([stats, wins])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Data(Dataset):
    def __init__(self, data, augmentation=False, normalization=False, with_team_info=False, with_win_info=False):
        super(Data, self).__init__()
        self.data = data
        self.ids_to_keys = {i: k for i, k in enumerate(data.keys())}
        self.augmentation = augmentation
        self.normalization = normalization
        self.with_team_info = with_team_info
        self.with_win_info = with_win_info

    def __getitem__(self, index):
        data = self.data[self.ids_to_keys[index]]
        stats = data['Stats']
        votes = data['Brownlow Votes']
        if self.normalization:
            stats = normalize_data(stats)
        if self.with_team_info:
            stats = add_team_info(stats, data['Teams'])
        if self.with_win_info:
            stats = add_win_info(stats, data['Home Team'], data['Teams'], data['Margin'])
        if self.augmentation:
            stats, votes = data_augmentation(stats, votes)
        votes_idx = np.argmax(votes)
        votes_dis = softmax(votes)

        return stats, votes_idx, votes_dis

    def __len__(self):
        return len(self.data)

#
# # test Data class
# dataset = Data(train_data)
# d = dataset[0]
# dataset = Data(train_data, augmentation=True)
# d = dataset[0]
# dataset = Data(train_data, True, True, True, True)
# d = dataset[0]

augmentation = True
normalization = True
with_team_info = True
with_win_info = True
batch_size = 128
train_loader = DataLoader(Data(train_data, augmentation=augmentation, normalization=normalization, with_team_info=with_team_info, with_win_info=with_win_info),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)
val_loader = DataLoader(Data(val_data, augmentation=augmentation, normalization=normalization, with_team_info=with_team_info, with_win_info=with_win_info),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True)
test_loader = DataLoader(Data(test_data, augmentation=False, normalization=normalization, with_team_info=with_team_info, with_win_info=with_win_info),
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=True)


class MLP(nn.Module):
    def __init__(self, input_dim=44*21, out_dim=44, p=0.4):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(p),
                                    nn.Linear(1024, out_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out


class CONV(nn.Module):
    def __init__(self, in_channels=21):
        super(CONV, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(in_channels, 64, 3, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 64, 3, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 128, 3, 2),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 256, 3, 2),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(256))
        self.fc = nn.Linear(256*9, 44)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        out = self.fc(x.view(x.size(0), -1))
        return out


class ATT(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super(ATT, self).__init__()
        self.K = nn.Conv1d(input_dim, inner_dim, 1, 1)
        self.Q = nn.Conv1d(input_dim, inner_dim, 1, 1)
        self.V = nn.Conv1d(input_dim, inner_dim, 1, 1)
        self.out = nn.Sequential(nn.Conv1d(inner_dim, output_dim, 1, 1),
                                 nn.BatchNorm1d(output_dim))
        if input_dim == output_dim:
            self.identity = lambda x: x
        else:
            self.identity = nn.Sequential(nn.Conv1d(input_dim, output_dim, 1, 1),
                                          nn.BatchNorm1d(output_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        s = torch.matmul(q, k.transpose(1, 2))
        s = F.softmax(s, -1)
        output = self.identity(x) + self.out(torch.matmul(s, v))
        output = F.sigmoid(output)
        return output


class TRANSFORMER(nn.Module):
    def __init__(self, input_dim=21, ):
        super(TRANSFORMER, self).__init__()
        self.layers = nn.Sequential(ATT(input_dim, 32, 64),
                                    ATT(64, 64, 128),
                                    ATT(128, 64, 128),
                                    ATT(128, 64, 128))
        self.fc = nn.Linear(44*128, 44)

    def forward(self, x):
        return self.fc(self.layers(x.transpose(1, 2)).view(x.size(0), -1))


def build_network(network_choice, with_win_info, with_team_info):
    feature_dim = 21
    if with_win_info:
        feature_dim += 1
    if with_team_info:
        feature_dim += 1
    if network_choice == 'mlp':
        return MLP(44*feature_dim)
    elif network_choice == 'conv':
        return CONV(feature_dim)
    elif network_choice == 'transformer':
        return TRANSFORMER(feature_dim)



def loss_function(output, target, target_dis=None):
    T = 1  # hyper-parameter
    alpha = 0.5  # hyper-parameter
    if target_dis is None:
        return nn.CrossEntropyLoss()(output, target)
    else:
        return nn.KLDivLoss()(F.log_softmax(output/T, dim=1),
                             F.softmax(target_dis/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(output, target) * (1. - alpha)


def metric_function(output, target, topk=1):
    """
    The accuracy of model that the top $k$ predictions being the player who got 3 votes
    :param output:
    :param target:
    :param topk:
    :return:
    """
    topks, topk_idxes = torch.topk(output, topk, dim=1)
    count = 0
    for p, t in zip(topk_idxes, target):
        if t in p:
            count += 1
    acc = count * 1.0 / len(target)
    return acc


def metric_function2(output, target_dist):
    """
    The accuracy of model that the prediction belongs to 1 of 3 players who got votes (1, 2, or 3)
    :param output: output of model
    :param target_dist: target distributions
    :return:
    """
    topks, topk_idxes = torch.topk(target_dist, 3, dim=1)
    _, p_topk_idxes = torch.topk(output, 1, dim=1)
    count = 0
    for p, t in zip(p_topk_idxes, topk_idxes):
        if p in t:
            count += 1
    acc = count * 1.0 / target_dist.size(0)

    return acc


def train(epoch, model, optimizer, dataloader, distill=False):
    model.train()

    for idx, (data, target, target_dis) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target, target_dis = data.float().to(device), target.to(device), target_dis.float().to(device)
        output = model(data)
        if distill:
            loss = loss_function(output, target, target_dis)
        else:
            loss = loss_function(output, target)

        loss.backward()
        optimizer.step()
        if idx % 20 == 0:
            print(f'Epoch {epoch} batch {idx}: loss {loss.item()}')
    return


def evaluate(epoch, model, dataloader, is_test=False):
    model.eval()
    top1 = 0
    top3 = 0
    top5 = 0
    acc3 = 0
    for idx, (data, target, target_dist) in enumerate(dataloader):
        data, target, target_dist = data.float().to(device), target.to(device), target_dist.to(device)
        output = model(data)
        top1 += metric_function(output, target, topk=1) * len(target)
        top3 += metric_function(output, target, topk=3) * len(target)
        top5 += metric_function(output, target, topk=5) * len(target)
        acc3 += metric_function2(output, target_dist) * len(target)
        # if idx % 10 == 0:
        #     print(f'Eval {epoch} batch {idx}: Acc {accuracy}')

    num_samples = len(dataloader.dataset)
    print(f'Test/Eval epoch {epoch} top1 is {top1/num_samples}, top3: {top3/num_samples}, top5: {top5/num_samples} | Acc3: {acc3/num_samples}')
    return top1


distill = False
network_type = 'transformer'  # hyper-parameter
lr = 0.01 #hyper-param
momentum = 0.9 #hyper-param
wd = 1e-4 #hyper-param
lr_steps = [10, 20, 25] #hyper-param
epochs = 30 #hyper-param
network = build_network(network_type, with_win_info, with_team_info)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, gamma=0.1)


best_acc = 0
for epoch in range(30):
    lr_scheduler.step(epoch)
    train(epoch, network, optimizer, train_loader, distill=distill)
    mean_acc = evaluate(epoch, network, val_loader)
    if (epoch + 1) % 5 == 0:
        torch.save(network.state_dict(), f'epoch{epoch}.pth')
    if best_acc < mean_acc:
        best_acc = mean_acc
        torch.save(network.state_dict(), 'best.pth')

network.load_state_dict(torch.load('best.pth'))
evaluate(-1, network, test_loader, True)

