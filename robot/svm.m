% 生成样本数据
rng(1); % 设置随机数种子以确保结果可复现
N = 50; % 样本数量
X = [randn(N,2)-1; randn(N,2)+1]; % 生成两类样本，分布在(-1,-1)和(1,1)附近
Y = [ones(N,1); -ones(N,1)]; % 设置类标签，分别为1和-1

% 定义优化问题
H = (Y * Y') .* (X * X');
f = -ones(N * 2, 1);
A = [];
b = [];
Aeq = Y';
beq = 0;
lb = zeros(N * 2, 1);
ub = [];

% 使用优化工具箱求解
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub);

% 计算权重向量和偏差项
w = (alpha .* Y)' * X;
b = mean(Y - X * w');

% 绘制决策边界
figure;
h = 0.02; % 网格步长
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
scores = (xGrid * w' + b)';
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');
hold on;

% 绘制原始样本
plot(X(Y==1,1), X(Y==1,2), 'ro', 'MarkerSize', 8); % 类别为1的样本用红色圆圈表示
plot(X(Y==-1,1), X(Y==-1,2), 'b*', 'MarkerSize', 8); % 类别为-1的样本用蓝色星号表示

title('SVM 决策边界');
xlabel('特征1');
ylabel('特征2');
legend('Decision Boundary', 'Class 1', 'Class -1');
hold off;
