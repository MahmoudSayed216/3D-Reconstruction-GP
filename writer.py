class Writer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.file_path = base_path+"/logs.txt"
        self.file = open(self.file_path, "+a")
    def save_configs(self, cfg):
        pass
    def add_line(self, line):
        self.file.write(line)
    
    def add_scaler(self, tag: str, step: int, value):

        line = f"step {step}: {tag} = {value}\n"
        self.file.write(line)

    def close(self):
        self.file.close()
    