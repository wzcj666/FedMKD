import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd = loss_kd.mean() if reduce else loss_kd
    return loss_kd * (temperature ** 2)


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    _, class_num = logits_teacher.shape
    pred_s = F.softmax(logits_student / temperature, dim=1)
    pred_t = F.softmax(logits_teacher / temperature, dim=1)
    s_matrix = torch.mm(pred_s.t(), pred_s)
    t_matrix = torch.mm(pred_t.t(), pred_t)
    diff = (t_matrix - s_matrix) ** 2
    return diff.sum() / class_num if reduce else diff / class_num


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, _ = logits_teacher.shape
    pred_s = F.softmax(logits_student / temperature, dim=1)
    pred_t = F.softmax(logits_teacher / temperature, dim=1)
    s_matrix = torch.mm(pred_s, pred_s.t())
    t_matrix = torch.mm(pred_t, pred_t.t())
    diff = (t_matrix - s_matrix) ** 2
    return diff.sum() / batch_size if reduce else diff / batch_size


def _multi_temp_loss(loss_fn, student_logits, teacher_logits, temps, weight, mask=None, reduce=True):
    total_loss = 0
    for t in temps:
        loss_val = loss_fn(student_logits, teacher_logits, t, reduce=False)
        if mask is not None:
            loss_val = (loss_val * mask).mean()
        elif reduce:
            loss_val = loss_val.mean()
        total_loss += loss_val
    return total_loss * weight


class clientMKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.temperature = 4.0
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = 0.5

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            W_h = nn.Linear(args.feature_dim, args.feature_dim, bias=False).to(self.device)
            save_item(W_h, self.role, 'W_h', self.save_folder_name)
            global_model = load_item('Server', 'global_model', self.save_folder_name)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.sample_per_class = torch.zeros(self.num_classes)
        for _, y in self.load_train_data():
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []


    def train(self):
        trainloader = self.load_train_data()

        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        optimizer_W = torch.optim.SGD(W_h.parameters(), lr=self.learning_rate)

        model.train()
        start_time = time.time()

        temps = [self.temperature, 2.0, 3.0, 5.0, 6.0]
        max_local_epochs = np.random.randint(1, self.local_epochs // 2) if self.train_slow else self.local_epochs

        for step in range(max_local_epochs):
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, use_cuda=True)

                logits_s_w = model(x)
                logits_s_s = model(x)

                with torch.no_grad():
                    logits_t_w = global_model(x)
                    logits_t_s = global_model(x)

                pred_t_w = F.softmax(logits_t_w.detach(), dim=1)
                confidence, pseudo_labels = pred_t_w.max(dim=1)
                conf_thresh = np.percentile(confidence.cpu().numpy(), 50)
                mask = confidence.le(conf_thresh).bool()

                class_conf = pred_t_w.sum(dim=0).detach()
                class_thresh = np.percentile(class_conf.cpu().numpy(), 50)
                class_mask = class_conf.le(class_thresh).bool()

                loss_kd_strong = _multi_temp_loss(kd_loss, logits_s_s, logits_t_s, temps, self.kd_loss_weight)
                loss_kd_weak_g = _multi_temp_loss(kd_loss, logits_t_w, logits_s_w, temps, self.kd_loss_weight, mask)
                loss_kd_strong_g = _multi_temp_loss(kd_loss, logits_t_s, logits_s_s, temps, self.kd_loss_weight)

                loss_cc_weak = _multi_temp_loss(cc_loss, logits_s_w, logits_t_w, temps, self.kd_loss_weight, class_mask)
                loss_cc_strong = _multi_temp_loss(cc_loss, logits_s_s, logits_t_s, temps, self.kd_loss_weight)

                loss_bc_weak = _multi_temp_loss(bc_loss, logits_s_w, logits_t_w, temps, self.kd_loss_weight, mask)
                loss_bc_strong = _multi_temp_loss(bc_loss, logits_s_s, logits_t_s, temps, self.kd_loss_weight)

                rep = model.base(x)
                rep_g = global_model.base(x)
                output = model.head(rep)
                output_g = global_model.head(rep_g)

                CE_loss = lam * self.loss(output, y_a) + (1 - lam) * self.loss(output, y_b)
                CE_loss_g = lam * self.loss(output_g, y_a) + (1 - lam) * self.loss(output_g, y_b)

                L_h = self.MSE(rep, W_h(rep_g)) * 0.5
                L_h_g = self.MSE(rep, W_h(rep_g)) * 0.5

                loss = CE_loss + loss_kd_strong + loss_cc_strong + loss_bc_strong + L_h
                loss_g = CE_loss_g + loss_kd_weak_g + loss_kd_strong_g + loss_cc_weak + loss_cc_strong + loss_bc_weak + loss_bc_strong + L_h_g

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                optimizer_W.zero_grad()

                loss.backward(retain_graph=True)
                loss_g.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(W_h.parameters(), 10)

                optimizer.step()
                optimizer_g.step()
                optimizer_W.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def _test(self, model_name):
        testloader = self.load_test_data()
        model = load_item(self.role, model_name, self.save_folder_name)
        model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                correct += (torch.argmax(output, dim=1) == y).sum().item()
                total += y.size(0)

        return correct, total


    def test_metricsgl(self):
        return self._test('global_model')


    def test_metrics(self):
        return self._test('model')
