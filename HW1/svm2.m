% 生成样本数据
rng(1); % 设置随机数种子以确保结果可复现
N = 50; % 样本数量
X = [randn(N,2)-1; randn(N,2)+1]; % 生成两类样本，分布在(-1,-1)和(1,1)附近
Y = [ones(N,1); -ones(N,1)]; % 设置类标签，分别为1和-1

% 初始化参数
w = randn(1, size(X, 2));
b = randn();

% 设置训练参数
maxIter = 100;
learningRate = 0.1;

% 训练迭代
for iter = 1:maxIter
    % 计算预测值
    predictions = sign(X * w' + b);
    
    % 更新参数
    w = w + learningRate * (Y' * X - Y' * predictions * X) / size(X, 1);
    b = b + learningRate * mean(Y - predictions);
end

% 绘制决策边界
figure;
scatter(X(Y==1,1), X(Y==1,2), 'ro'); hold on; % 类别为1的样本用红色圆圈表示
scatter(X(Y==-1,1), X(Y==-1,2), 'b*'); % 类别为-1的样本用蓝色星号表示

% 找到 x1 的范围
x1range = [min(X(:,1)), max(X(:,1))];

% 根据 x1 的范围计算 x2
x2 = (-b - w(1)*x1range') / w(2); % 注意转置

% 绘制决策边界
plot(x1range, x2, 'k', 'LineWidth', 2);
title('SVM 决策边界');
xlabel('特征1');
ylabel('特征2');
legend('Class 1', 'Class -1', 'Decision Boundary');
