import cv2
import numpy as np 
import os

class DarknetDataset:
    def __init__(self, data_root, is_test=False, transform=None, target_transform=None):

        if is_test:
            self.data_path = data_root + '/images/validation'
            self.labels_path = data_root + '/labels/validation'
        else:
            self.data_path = data_root + '/images/train'
            self.labels_path = data_root + '/labels/train'

        label_file_name = data_root + '/labels/labels.txt'

        if os.path.isfile(label_file_name):
            classes = []
            # classes should be a line-separated list
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())
                    
        self.class_names = tuple(classes)
    
        self.transform = transform
        self.target_transforms = target_transform
        self.images_data, self.bboxes, self.labels = list(), list(), list()

        for file in os.listdir(self.data_path):
            img_name = os.fsdecode(file)
            if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                img_path = os.path.join(self.data_path, img_name)
                img = self._read_image(img_path)
                self.images_data.append(img)
        
        for file in os.listdir(self.labels_path):
            label_name = os.fsdecode(file)
            if label_name.endswith('.txt'):
                label_path = os.path.join(self.labels_path, label_name)
                bboxes_idx, labels_idx = self._get_annot_each_label(label_path)
                self.bboxes.append(bboxes_idx)
                self.labels.append(labels_idx)
        
        # print(f'[INFO] Num images data -> validation:{is_test} = {len(self.images_data)}')
        # print(f'[INFO] Num bboxes data -> validation:{is_test} = {len(self.bboxes)}')
        # print(f'[INFO] Num labels data -> validation:{is_test} = {len(self.labels)}')
        assert len(self.images_data) == len(self.bboxes) == len(self.labels)

    def _get_annot_each_label(self, label_path):
        f = open(label_path, "r")
        annotations = f.readlines()
        f.close()
        bboxes_idx, labels_idx = list(), list()
        for annot in annotations:
            annot_data = annot.split(' ')
            clsID = int(annot_data[0].split()[0])
            # bbox format [x_center, y_center, w, h]
            bbox_0 = float(annot_data[1].split()[0])
            bbox_1 = float(annot_data[2].split()[0])
            bbox_2 = float(annot_data[3].split()[0])
            bbox_3 = float(annot_data[4].split()[0])
            # convert to [x1,y1,x2,y2]
            x1_coord = (bbox_0 - bbox_2/2)*512
            x2_coord = (bbox_0 + bbox_2/2)*512
            y1_coord = (bbox_1 - bbox_3/2)*512
            y2_coord = (bbox_1 + bbox_3/2)*512
            bbox = [x1_coord, y1_coord, x2_coord, y2_coord]
            bboxes_idx.append(bbox)
            labels_idx.append(clsID)
        return (np.array(bboxes_idx, dtype=np.float32),
                np.array(labels_idx, dtype=np.int64))

    def _read_image(self, img_path):
        
        if img_path is None:
            raise IOError('failed to load ' + img_path)
            
        img = cv2.imread(str(img_path))
        
        if img is None or img.size == 0:
            raise IOError('failed to load ' + str(img_path))
            
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        img_sample = self.images_data[index]
        boxes_sample = self.bboxes[index]
        labels_sample = self.labels[index]
        assert len(boxes_sample) == len(labels_sample)

        if self.transform:
            img_sample, boxes_sample, labels_sample = self.transform(img_sample, boxes_sample, labels_sample)
        if self.target_transforms:
            boxes_sample, labels_sample = self.target_transforms(boxes_sample, labels_sample)
            
        return img_sample, boxes_sample, labels_sample

    def __len__(self):
        return len(self.images_data)

