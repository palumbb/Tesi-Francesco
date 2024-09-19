from torch.optim import Adam, Optimizer, SGD

class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, grads, lr, momentum, weight_decay):
        super().__init__(
            grads, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        """Implement the custom step function fo SCAFFOLD."""
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])
