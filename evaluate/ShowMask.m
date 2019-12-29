
function [] = ShowMask(file_path, file_name, bNum, nBlocks)

Data_ = load(strcat(file_path, 'result/', file_name));
weis_out = squeeze(Data_.weis_out);
size(weis_out)

img_num = bNum * (nBlocks+2);

names = strsplit(file_name, '.');

w = 0.333;
h = 0.199;

figure(1)
for i = 1:(nBlocks+2)
    for j = 1:bNum
        num = (i-1)*(bNum) + j;
        colormap jet
        imagesc(squeeze(weis_out(num,:,:)));
        set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
        mask_name = char(strcat(file_path, 'mask/', names(1), '_', num2str(i), '_', num2str(j)))
        saveas(gcf, mask_name, 'png')
        % saveas(gcf,mask_name,'pdf')
        % print('-dtiffn', mask_name);%, '-r1300'
    end
end

end