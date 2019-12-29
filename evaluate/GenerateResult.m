function [] = GenerateResult(Path_T, Path_R)

names_T = dir(strcat(Path_T, '*.mat'));
names_R = dir(strcat(Path_R, '*.mat'));

d_len = length(names_T);

psnr_sum = 0;
rmse_sum = 0;
sam_sum = 0;
ssim_sum = 0;

result = zeros(d_len+1, 4);
for i=1:d_len
    name = names_T(i).name
    Data_T = load(strcat(Path_T, name));
    name = names_R(i).name
    Data_R = load(strcat(Path_R, name));

    T = Data_T.rad;
    R = Data_R.rad;
    T = permute(T, [3,1,2]);
    
    T = T/4095;
    % R = R/4095;
    R = double(R);
    R(R>1) = 1;
    R(R<0) = 0;
    size(T), size(R), class(T), class(R)
    [max(max(max(T))),max(max(max(R))), min(min(min(T))),min(min(min(R)))]
    
    [~,m,n] = size(T);
    B_mse(:,:) = sqrt(mean((T-R).^2,1))*255.0;
    max(max(max(B_mse)))
    
    % 可视化mse图
    figure(1);
    imagesc(B_mse,[0,15]);
    colorbar;
    print('-dpng',strcat(Path_R, 'mse_', num2str(i), '.png'));

    % 可视化某个波段
    %[Path_T 'T_' num2str(i) '.png']
    %imwrite(squeeze(T(31,:,:)), [Path_R 'T_' num2str(i) '.png']);
    %imwrite(squeeze(R(31,:,:)), [Path_R 'R_' num2str(i) '.png']);
    
    [q,m,n] = size(T);
    T_2D = reshape(T,q,m*n);
    R_2D = reshape(R,q,m*n);
    [PSNR,RMSE] = Evaluate(T_2D,R_2D);
    % PSNR = psnr(permute(R, [2,3,1]), permute(T, [2,3,1]));
    
    SAM = SpectAngMapper(permute(T,[2,3,1]),permute(R,[2,3,1]));
    %SSIM = GetSSIMofHSI(double(T_2D),double(R_2D),m,n);
    SSIM = ssim(permute(R,[2,3,1]), permute(T,[2,3,1]));
   
    fprintf('%s  :%.2f , %.2f , %.2f , %.4f.\n',name(1:end-4),RMSE,PSNR,SAM,SSIM);
    psnr_sum = psnr_sum+PSNR;
    rmse_sum = rmse_sum+RMSE;
    sam_sum = sam_sum+SAM;
    ssim_sum = ssim_sum+SSIM;
    result(i,:) = [RMSE, PSNR, SAM, SSIM];
end
result(d_len+1,:) = [rmse_sum/d_len, psnr_sum/d_len, sam_sum/d_len, ssim_sum/d_len];
fprintf('The average of result is :%.2f , %.2f , %.2f , %.4f.\n',rmse_sum/d_len,psnr_sum/d_len,sam_sum/d_len,ssim_sum/d_len);

colnames = {'RMSE', 'PSNR', 'SAM', 'SSIM'};
xlswrite(strcat(Path_R, 'result.xls'), result);

end
