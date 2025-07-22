from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import joblib

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if outputs.dim() == 2:
                    outputs = outputs.unsqueeze(-1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if outputs.dim() == 2:
                            outputs = outputs.unsqueeze(-1)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if outputs.dim() == 2:
                        outputs = outputs.unsqueeze(-1)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # --- ★★★ 핵심 수정 사항 2 ★★★ ---
        # 학습 데이터의 스케일러를 파일로 저장
        scaler_path = os.path.join(path, 'scaler.gz')
        joblib.dump(train_data.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        # ----------------------------------
        return self.model

    # exp_long_term_forecasting.py의 test 함수

    def test(self, setting, test=0):
      test_data, test_loader = self._get_data(flag='test')
      if test:
          print('loading model')
          self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

      preds = []
      trues = []
      attentions = []

      # 1. 시각화(PDF) 저장을 위한 폴더 경로 설정
      pdf_folder_path = './test_results/' + setting + '/'
      if not os.path.exists(pdf_folder_path):
          os.makedirs(pdf_folder_path)

      self.model.eval()
      with torch.no_grad():
          for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)
              batch_x_mark = batch_x_mark.float().to(self.device)
              batch_y_mark = batch_y_mark.float().to(self.device)

              dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
              dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

              if self.args.output_attention:
                  outputs, *attention = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
              else:
                  outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

              if outputs.dim() == 2:
                  outputs = outputs.unsqueeze(-1)

              f_dim = -1 if self.args.features == 'MS' else 0
              outputs = outputs[:, -self.args.pred_len:, :]
              batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
              outputs = outputs.detach().cpu().numpy()
              batch_y = batch_y.detach().cpu().numpy()

              if self.args.output_attention:
                  attentions.append(attention[0][0].detach().cpu().numpy())

              if test_data.scale and self.args.inverse:
                  unscaled_outputs_list = []
                  unscaled_batch_y_list = []

                  for j in range(outputs.shape[0]):
                      output_sample = outputs[j]
                      true_sample = batch_y[j]

                      if self.args.features == 'MS':
                          num_features = test_data.scaler.n_features_in_
                          output_sample = np.tile(output_sample, (1, num_features))

                      unscaled_output = test_data.inverse_transform(output_sample)
                      unscaled_outputs_list.append(unscaled_output)

                      unscaled_true = test_data.inverse_transform(true_sample)
                      unscaled_batch_y_list.append(unscaled_true)

                  outputs = np.stack(unscaled_outputs_list, axis=0)
                  batch_y = np.stack(unscaled_batch_y_list, axis=0)

              outputs = outputs[:, :, f_dim:]
              batch_y = batch_y[:, :, f_dim:]

              pred = outputs
              true = batch_y

              preds.append(pred)
              trues.append(true)

              if i % 20 == 0:
                  input = batch_x.detach().cpu().numpy()
                  if test_data.scale and self.args.inverse:
                      shape = input.shape
                      input = test_data.inverse_transform(input.reshape(-1, input.shape[-1])).reshape(shape)
                  gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                  pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                  # PDF 저장 경로를 'pdf_folder_path'로 지정
                  visual(gt, pd, os.path.join(pdf_folder_path, str(i) + '.pdf'))

      preds = np.concatenate(preds, axis=0)
      trues = np.concatenate(trues, axis=0)
      print('test shape:', preds.shape, trues.shape)

      # 2. 최종 결과(NPY, TXT) 저장을 위한 폴더 경로 설정
      results_folder_path = './results/' + setting + '/'
      if not os.path.exists(results_folder_path):
          os.makedirs(results_folder_path)

      if self.args.output_attention:
          attentions = np.concatenate(attentions, axis=0)
          print('attention shape:', attentions.shape)
          np.save(os.path.join(results_folder_path, 'attention.npy'), attentions)

      mae, mse, rmse, mape, mspe = metric(preds, trues)
      print('mse:{}, mae:{}'.format(mse, mae))

      # 결과 저장 경로를 'results_folder_path'로 지정
      with open(os.path.join(results_folder_path, "result_metrics.txt"), 'a') as f:
          f.write(setting + "  \n")
          f.write('mse:{}, mae:{}'.format(mse, mae))
          f.write('\n')
          f.write('\n')

      np.save(os.path.join(results_folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
      np.save(os.path.join(results_folder_path, 'pred.npy'), preds)
      np.save(os.path.join(results_folder_path, 'true.npy'), trues)

      return