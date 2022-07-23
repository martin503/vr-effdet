class Config:
    def __init__(self, epochs=100, val_gap=20,
                 batch_size=512, learning_rate=1e-3, weight_decay=1e-5,
                 edge_ratio=1):
        # Training
        self.epochs = epochs
        self.val_gap = val_gap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Ratios
        self.edge_ratio = edge_ratio
