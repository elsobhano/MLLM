from models.PreTrain import PreTrain




if __name__ == '__main__':
    model = PreTrain()
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")