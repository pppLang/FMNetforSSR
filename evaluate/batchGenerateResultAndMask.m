clear;clc;

Path_T = '/data0/langzhiqiang/data/test_gt/';

root_path = ["/data0/langzhiqiang/FMNetworkforSSR/logs/FMNet_original_3_2_0.0001_[20, 40, 60, 80]_0.5_64/"];

for i = 1:length(root_path)
    
    
    Path_R = root_path(i)

    % 先计算四项指标，以及RMSE map图
    % GenerateResult(Path_T, Path_R);

    % 保存mask图
    file_name_list = dir(strcat(Path_R, 'result/', '*.mat'));
    for j=1:length(file_name_list)
       file_name = file_name_list(j).name;
       ShowMask(Path_R, file_name, double(3), double(2));
    end
    break
end
