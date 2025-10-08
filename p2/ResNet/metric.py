import torch
import numpy as np
import json
class MetricController():
    def __init__(self):
        self.state_dic: dict[str,list[np.ndarray]] = {} #[batch , batch ~]
    def reset(self):
        self.state_dic = {}
    def add(self,metric_name:str , value:list[torch.tensor])->None:
        """
        Args:
            metric_name: mse or mae ~
            values: 1 epoch losses list
        """
        if(value[0].shape != torch.tensor(1).shape):
            print('value\' component shpae is', value.shpae , ' it should be ',torch.tensor(1).shape)
            return
        if metric_name not in self.state_dic.keys():

            self.state_dic[metric_name] = value.detach().cpu().numpy()
        else:
            self.state_dic[metric_name].append(value.detach().cpu().numpy())
    def check(self,metric_name):
        if metric_name not in self.state_dic.keys():
            print(metric_name,'is not in ',self.state_dic.keys())
            return False
        return True
    def recent_mean(self,metric_name:str)->float:
        """
        recent epoch mean loss
        """
        if self.check(metric_name):
            arr = self.state_dic[-1]
            return float(arr.mean())
        else:
            print('error')
            return
    def recent_std(self,metric_name:str)->float:
        if self.check(metric_name):
            arr = self.state_dic[-1]
            return float(arr.std())
        else:
            print('error')
            return
    def show_keys(self):
        print('keys in metriccontroller:',self.state_dic.keys())
    def plot(self,metric_name,path):
        pass
    def save(self,path,metrci_name):
        if self.check(metrci_name):
            temp = [arr.tolist() for arr in self.state_dic[metrci_name]]
            with open(path,'w',encoding='utf-8') as f:
                json.dump(temp,f,indent =4)
                print('trainning log saved at',path)
        


        
