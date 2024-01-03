import matplotlib.pyplot as plt

with open("graph_pirl_resnet", encoding="utf-8") as f:
    data = f.readlines()

train_loss_list = []
test_loss_list = []
for item in data:
    if len(item) > 3:
        info = item.split("-")[1].strip()
    else:
        continue
    #info = item
    print(info)
    #if info.startswith("Train set"):
    #    print(info)
    #    train_loss = info.split(",")[1].split()[-1][1:-2]
    #    train_loss_list.append(train_loss)
    if info.startswith("Test set"):
        print(info)
        test_loss = info.split(",")[1].split()[-1][1:-2]
        test_loss_list.append(test_loss)

train_loss_list = [float(item) for item in train_loss_list]
test_loss_list = [float(item) for item in test_loss_list]
print(train_loss_list)
print(test_loss_list)


x_values = list(range(len(train_loss_list)))

plt.plot(x_values, train_loss_list, marker='o', linestyle='-')


plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Resnet18 + PIRL Finetuning')


x_values = list(range(len(test_loss_list)))

plt.plot(x_values, test_loss_list, marker='o', linestyle='-')



plt.grid(True)
plt.tight_layout()

plt.savefig("graph_pirl_resnet.png")


