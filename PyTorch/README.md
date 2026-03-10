# PyTorch Gradients and Backprop (Manual + Mermaid)

This mini-lecture shows how PyTorch autograd matches manual chain-rule math for two files:
`gradient.py` and `backporpogation.py`.

## gradient.py

### Code summary
```
x = torch.rand(3, requires_grad=True)
y = x + 2
z = y * y * 2
z.backward(y)
```

### Manual chain-rule
Let:
- `y = x + 2`
- `z = 2 * y^2`

`z.backward(y)` supplies an external gradient `g = dL/dz = y`.

Then:
- `dz/dy = 4y`
- `dL/dy = dL/dz * dz/dy = y * 4y = 4y^2`
- `dy/dx = 1`
- `dL/dx = dL/dy * dy/dx = 4y^2 = 4(x+2)^2`

So `x.grad` should equal `4(x+2)^2` elementwise.

### Mermaid graph (chain rule)
```mermaid
flowchart LR
    x[x] --> y[y = x + 2]
    y --> z[z = 2 * y^2]
    g[dL/dz = y] --> z
    z --> dy[dL/dy = (dL/dz) * (dz/dy)]
    dy --> dx[dL/dx = dL/dy * dy/dx]
```

## backporpogation.py

### Code summary
```
x = 1.0
y = 2.0
w = 1.0 (requires_grad)

y_hat = w * x
loss = (y_hat - y)^2
loss.backward()
```

### Manual chain-rule
Let:
- `y_hat = w * x`
- `loss = (y_hat - y)^2`

Then:
- `dloss/dy_hat = 2(y_hat - y)`
- `dy_hat/dw = x`
- `dloss/dw = 2(y_hat - y) * x`

With `x = 1`, `y = 2`, `w = 1`:
- `y_hat = 1`
- `dloss/dw = 2(1 - 2) * 1 = -2`

So `w.grad` should print `-2`.

### Mermaid graph (chain rule)
```mermaid
flowchart LR
    w[w] --> yhat[y_hat = w * x]
    x[x] --> yhat
    yhat --> loss[loss = (y_hat - y)^2]
    y[y] --> loss
    loss --> grad[dloss/dw = 2(y_hat - y) * x]
```

## Notes
- `z.backward(y)` uses a vector-Jacobian product; the argument is the external gradient `dL/dz`.
- If you replace `z.backward(y)` with `z.sum().backward()`, the external gradient is `1`.
