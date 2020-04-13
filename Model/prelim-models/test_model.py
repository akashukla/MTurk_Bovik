# coding: utf-8
i=0
t_x = torch.Tensor(X_test)
t_y = torch.Tensor(y1_test)
d = data.TensorDataset(t_x, t_y)
dataloader_test = data.DataLoader(d, num_workers=mp.cpu_count())
#model4 =torch.load('testsave')
#model4.eval()
for inputs, labels  in dataloader_test:
    i+=1
    if(i>10):
        break
    inputs=inputs.to(device)
    labels=labels.to(device)
    print(labels)
    outputs = model(inputs)
    print(outputs)
    
