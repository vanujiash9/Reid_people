import os
import yaml

class YOLODataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare_data_yaml(self):
        """Tạo file data.yaml mà YOLOv8 yêu cầu"""
        data_info = {
            'path': os.path.abspath(self.cfg['data']['root_dir']),
            'train': 'images/train',
            'val': 'images/train',  # Tận dụng tập train để giám sát quá trình học
            'nc': 1,
            'names': ['person']
        }
        
        save_path = os.path.join("config", self.cfg['data']['yaml_name'])
        # Đảm bảo ghi file bằng UTF-8
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_info, f, default_flow_style=False)
            
        print(f"[INFO] YOLO Data YAML created at: {save_path}")
        return save_path