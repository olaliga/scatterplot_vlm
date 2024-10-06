import pandas as pd
import numpy as np
import random, os, cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.gridspec as gridspec



class BasicData:
    def __init__(self, filename, params):
        self.filename = filename
        self.params = params
        self.model_fcn = lambda x:self.params["coef1"]*(x**self.params["model_curvature"]) + self.params["coef0"]
        
        self.model_table = self.create_table(relation = self.params["relation"])
        self.silicon_table = self.create_table(num_pts = self.params["data_counts"], rand = True)

        self.figure = self.create_figure()
        self.caption = self.create_caption()

    def create_table(self, num_pts = 7, rand = False, relation = None):
        x = np.linspace(self.params["x_min"], self.params["x_max"], num_pts)
        y = None

        if rand : 
            noise = np.random.normal(0, self.params["silicon_spreadness"], num_pts)
            y = self.model_fcn(x) + noise
        else: 
            if relation == "higher than":
                y = self.model_fcn(x) + self.params["coef0"]*0.7 + 1
            elif relation == "slightly higher than":
                y = self.model_fcn(x) + self.params["coef0"]*0.3
            elif relation == "overlapping with":
                y = self.model_fcn(x)
            if relation == "lower than":
                y = self.model_fcn(x) - self.params["coef0"]*0.7 - 1
            elif relation == "slightly lower than":
                y = self.model_fcn(x) - self.params["coef0"]*0.3

        return pd.DataFrame({self.params["x_axis"] : x, self.params["y_axis"] : y})

    def create_figure(self, append_grid = True, control_size = True):
        if control_size:
            fig = plt.figure(figsize = (4,3))
            gs = gridspec.GridSpec(22, 22)
            ax1 = fig.add_subplot(gs[3:20, 2:20])
        else:
            fig, ax1 = plt.subplots()
        
        ax1.scatter(x = self.silicon_table[self.params["x_axis"]],
                    y = self.silicon_table[self.params["y_axis"]],
                    color = "b", facecolors = "none", marker = "s", label = "silicon")
        ax1.scatter(x = self.model_table[self.params["x_axis"]],
                    y = self.model_table[self.params["y_axis"]],
                    color = "r", marker = "d", label = "Model")
        

        ax1.set_xlabel(self.params["x_axis"])
        ax1.set_ylabel(self.params["y_axis"])
        ax1.set_title(self.params["title"])
        min_y = min(-2,min(self.silicon_table[self.params["y_axis"]]), min(self.model_table[self.params["y_axis"]])) - 1
        max_y = max(4, max(self.silicon_table[self.params["y_axis"]]), max(self.model_table[self.params["y_axis"]])) + 1
        ax1.set_ylim(min_y, max_y) 

        if append_grid:
            ax1.grid(color = "g", linestyle = "--", linewidth = 0.5, axis = "both")
        img = plt.gcf()
        plt.close()
        return img
    
    def create_caption(self):
        y_axis = self.params["y_axis"]
        x_axis = self.params["x_axis"]
        relation = self.params["relation"]

        return [f"The model is {relation} the silicon data points"]
    
    def get_save_files(self):
        return {"caption" : pd.DataFrame({"caption" : self.caption, "filename" : self.filename}),
                f"{self.filename}.png" : self.figure
                }
    
class DataSetGenerator:
    def __init__(self, num_figure, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        #DB control
        self.relation_list =["overlapping with"] # ["higher than", "slightly higher than", "overlapping with", "lower than", "slightly lower than"]
        self.model_curvature_max = 0.5
        self.silicon_spreadness_max = 0.01
        self.x_max_list = [0.7, 5, 10]
        self.x_min_list = [0, 3, 5]

        self.coef0_max = 3.0
        self.coef1_max = 0
        self.coef1_min = 0

        self.y_axis_lst = ["yvar1", "yvar2", "yvar3"]
        self.x_axis_lst = ["xvar1", "xvar2", "xvar3"]
        self.datacounts = 100
        self.datasets = []
        self.w_lst = [0.019, 0.023, 0.024]
        self.l_lst = [0.003, 0.005, 0.007, 0.009, 0.012]

        self.filename_lst = [f"testcase_{i}" for i in range(num_figure)]
        self.create_data()

    def create_data(self):
        for filename in self.filename_lst:
            data = BasicData(filename, params=self.extract_attr())
            self.datasets.append(data)

    def extract_attr(self):
        x_max = 0; x_min = 0
        while (x_max - x_min) <= 0:
            x_max = random.choice(self.x_max_list)
            x_min = random.choice(self.x_min_list)
        
        return {"data_counts" : self.datacounts,
                "model_curvature" : np.random.uniform(0.0, self.model_curvature_max),
                "silicon_spreadness" : np.random.uniform(0.1, self.silicon_spreadness_max),
                "relation" : random.choice(self.relation_list),
                "x_max" : x_max,
                "x_min" : x_min,
                "coef0" : np.random.uniform(0.0, self.coef0_max),
                "coef1" : np.random.uniform(self.coef1_min, self.coef1_max),
                "y_axis" : random.choice(self.y_axis_lst),
                "x_axis" : random.choice(self.x_axis_lst),
                "title" : f"xxxx W/L = {random.choice(self.w_lst)}/{random.choice(self.l_lst)}xxxx"
        }
    
    def save(self):
        caption_lst = []
        for data in self.datasets:
            files_dict = data.get_save_files()
            for filename, content in files_dict.items():
                if filename == "caption":
                    caption_lst.append(content)
                else :
                    content.savefig(f"{self.save_dir}/{filename}")
        pd.concat(caption_lst).to_csv(f"{self.save_dir}/caption.csv")