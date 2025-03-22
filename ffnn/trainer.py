class Trainer:
    def __init__(self, model, loss, d_loss, lr=0.01):
        self.model = model
        self.loss = loss
        self.d_loss = d_loss
        self.lr = lr
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.model.forward(X)
            loss_value = self.loss(y, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value}")