data = readmatrix('data_collection\forces_data.csv');
cd = -data(:,[1,4]);
cl = -data(:,[3,6]);
ltd = cl./cd;
duration = data(:,7);
MSE = data(:,8);
PCC =data(:,9);
R2 = data(:,10);
%%
figure(1)
indata = cd;
scatter(indata(:,1), indata(:,2),'b.')
hold on
plot([min(indata(:)),max(indata(:))], [min(indata(:)),max(indata(:))], ...
    'Color' ,[0.7,0.7,0.7])
axis equal

figure(2)
indata = cl;
scatter(indata(:,1), indata(:,2),'b.')
hold on
plot([min(indata(:)),max(indata(:))], [min(indata(:)),max(indata(:))], ...
    'Color' ,[0.7,0.7,0.7])
axis equal

figure(3)
indata = ltd;
scatter(indata(:,1), indata(:,2),'b.')
hold on
plot([min(indata(:)),max(indata(:))], [min(indata(:)),max(indata(:))], ...
    'Color' ,[0.7,0.7,0.7])
axis equal
%%
disp(['time: ' num2str(mean(duration)) '±' num2str(std(duration))])
disp(['MSE: ' num2str(mean(MSE)) '±' num2str(std(MSE))])
disp(['PCC: ' num2str(mean(PCC)) '±' num2str(std(PCC))])
disp(['R2: ' num2str(mean(R2)) '±' num2str(std(R2))])