from dataset import ResDataset
from model import ResNetRegressor
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from metric import MetricController
class Trainer():
    def __self__(self,device,config,
                checkpoint='microsoft/resnet-50'
                ):
        self.checkpoint = checkpoint
        self.device = device
        self.config  = config
        self.load_items()

    def load_items(self):
        self.model =ResNetRegressor(checkpoint=self.checkpoint)
        self.processor = self.model.processor
        dataset = ResDataset(img_dir=self.config['img_dir'] , label_dir= self.label_dir['label_dir'],preprocessor=self.processor,device = self.device)
        self.loss = torch.nn.MSELoss() ## 평가는 RMSE
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.config['lr'])
        self.dataloader = DataLoader(dataset,batch_size=self.config['batch'],shuffle=True)
        self.metriccontroller = MetricController()
    def train(self):
        print('train start\n')
        self.metriccontroller.reset()
        for _ in range(self.config['epoch']):
            losses = []
            for data in self.dataloader:
                self.optimizer.zero_grad()
                img = data['img']
                bbox = data['bbox']
                height = data['height']
                out = self.model(img,bbox)
                loss = self.loss(out,height)
                losses.append(loss)
                print(loss.shape)
                loss.bachward()
                self.optimizer.step()
            self.metriccontroller.add('MSE',losses)
            print('epoch:',_ ,' ,loss(mse) mean:',self.metriccontroller.recent_mean('MSE'),' , std:',self.metriccontroller.recent_std('MSE'),'\n')
        print('trainning end')
        # self.metriccontroller.plot('MSE')
        # self.metriccontroller.save(path = '~','MSE')

            
            

