% 自定义sigmoid函数
function y = custom_sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% 数据准备
% 假设有一个简单的二分类问题，输入特征为2维，输出为1维。

% 生成示例数据
X = [0 0; 0 1; 1 0; 1 1]; % 输入特征
Y = [0; 1; 1; 0]; % 输出标签

% 参数设置
input_size = 2; % 输入层节点数
hidden_size = 4; % 隐藏层节点数
output_size = 1; % 输出层节点数
learning_rate = 0.1; % 学习率
epochs = 10000; % 迭代次数

% 初始化权重和偏置
hidden_weights = randn(input_size, hidden_size); % 输入层到隐藏层的权重
hidden_bias = zeros(1, hidden_size); % 隐藏层的偏置
output_weights = randn(hidden_size, output_size); % 隐藏层到输出层的权重
output_bias = zeros(1, output_size); % 输出层的偏置

% 训练网络
for epoch = 1:epochs
    % 前向传播
    hidden_layer_input = X * hidden_weights + hidden_bias; % 隐藏层输入
    hidden_layer_output = custom_sigmoid(hidden_layer_input); % 隐藏层输出
    output_layer_input = hidden_layer_output * output_weights + output_bias; % 输出层输入
    predicted_output = custom_sigmoid(output_layer_input); % 输出层输出
    
    % 计算损失
    loss = 0.5 * sum((predicted_output - Y).^2);
    
    % 反向传播
    output_error = (predicted_output - Y) .* (predicted_output .* (1 - predicted_output)); % 输出层误差
    hidden_error = (output_error * output_weights') .* (hidden_layer_output .* (1 - hidden_layer_output)); % 隐藏层误差
    
    % 更新权重和偏置
    output_weights = output_weights - learning_rate * (hidden_layer_output' * output_error);
    output_bias = output_bias - learning_rate * sum(output_error);
    hidden_weights = hidden_weights - learning_rate * (X' * hidden_error);
    hidden_bias = hidden_bias - learning_rate * sum(hidden_error);
    
    % 输出当前损失
    if mod(epoch, 100) == 0
        disp(['Epoch: ', num2str(epoch), ', Loss: ', num2str(loss)]);
    end
end

% 预测
% 使用训练好的模型进行预测
hidden_layer_input = X * hidden_weights + hidden_bias;
hidden_layer_output = custom_sigmoid(hidden_layer_input);
output_layer_input = hidden_layer_output * output_weights + output_bias;
predicted_output = custom_sigmoid(output_layer_input);

disp('Predicted Output:');
disp(predicted_output);
