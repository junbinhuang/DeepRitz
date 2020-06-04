%% Read data from the Python outputs.
fid = fopen('lossData0.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData_itr = data{1,1}';
lossData = zeros(5,length(lossData_itr));
lossData(1,:) = data{1,2}';

fid = fopen('lossData1.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData(2,:) = data{1,2}';

fid = fopen('lossData2.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData(3,:) = data{1,2}';

fid = fopen('lossData3.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData(4,:) = data{1,2}';

fid = fopen('lossData4.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData(5,:) = data{1,2}';

%% Plot the error curve
% if size(lossData,2) > 500
%     lossData = lossData(:,10:10:end);
%     lossData_itr = lossData_itr(:,10:10:end);
% end
logLossData = log10(lossData);
lossData_err = mean(logLossData);
lossData_err_std = std(logLossData);
lossData_err1 = lossData_err-lossData_err_std;
lossData_err2 = lossData_err+lossData_err_std;
lossData_err = 10.^lossData_err;
lossData_err1 = 10.^lossData_err1;
lossData_err2 = 10.^lossData_err2;

figure
semilogy(lossData_itr,lossData_err,'b-','LineWidth',1.0)
hold on
XX = [lossData_itr, fliplr(lossData_itr)];
YY = [lossData_err1, fliplr(lossData_err2)];
theFill = fill(XX,YY,'b');
set(theFill,'facealpha',0.3,'edgecolor','b','edgealpha',0.0)

ylabel('Error','Interpreter','latex')
xlabel('Iterations','Interpreter','latex')
% ylim([0.005,1])
% legend({'No pre-training'},'Interpreter','latex')
set(gca,'ticklabelinterpreter','latex','fontsize',11)

%% Now we can start plotting figures.
fid=fopen('nSample.txt');
data = textscan(fid, '%d', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
numbers=cell2mat(data);

fid=fopen('boundaryNumber.txt');
data = textscan(fid, '%d', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
boundaryNumber=cell2mat(data);

fid=fopen('boundaryCoord.txt');
data = textscan(fid, '%f %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
bCoord=cell2mat(data);

nSample=numbers(1);

fid=fopen('Data.txt');
a = '%f ';
for i = 1:nSample-2
    a = [a,'%f '];
end
a = [a,'%f'];
data = textscan(fid, a, 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
totalData=cell2mat(data);

clear data numbers a

% Plot the boundary.
figure
hold on
axis equal
axis off

% Plot the contourf results!
plotData=totalData;
    
nScale=100;
nDomain=size(plotData,1)/nSample/3;

xArray=plotData(1:3:end,:);
yArray=plotData(2:3:end,:);
zArray=plotData(3:3:end,:);

xMin=min(xArray(:));xMax=max(xArray(:));
yMin=min(yArray(:));yMax=max(yArray(:));
zMin=min(zArray(:));zMax=max(zArray(:));
    
%% Set limits
zMin=0;
zMax=1;
%%

scale=linspace(zMin,zMax,nScale);

for i=1:nDomain        
    myContourf(xArray(nSample*(i-1)+1:nSample*i,:),...
                yArray(nSample*(i-1)+1:nSample*i,:),...
                zArray(nSample*(i-1)+1:nSample*i,:),scale)
end

xlim([xMin,xMax])
ylim([yMin,yMax])
% Colorbar limits
caxis([zMin,zMax])

boundaryCoord=bCoord;

%% Plot the boundary.
for i=1:length(boundaryNumber)
    coord=boundaryCoord(1:boundaryNumber(i),:);
    boundaryCoord=boundaryCoord(boundaryNumber(i)+1:end,:);

    plot(coord(:,1),coord(:,2),'k','LineWidth',1.5)
    if coord(1,1)~=coord(end,1) || coord(1,2)~=coord(end,2)
        plot(coord([1,end],1),coord([1,end],2),'k','LineWidth',1.5)
    end
end

clear plotData xArray yArray zArray scale nDomain xMin xMax yMin yMax zMin zMax...
     i fid coord

%% Some functions used:
function myContourf(x,y,z,scale)
%Used in visualization
    contourf(x,y,z,scale,'LineStyle','none');
    set(gca,'ticklabelinterpreter','latex','fontsize',11)
    colormap(jet);
    colorbar('ticklabelinterpreter','latex')
end