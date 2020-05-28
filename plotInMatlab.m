%% Read data from the Python outputs.
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

fid = fopen('lossData.txt');
data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
fclose(fid);
lossData_itr = data{1,1};
lossData_err = data{1,2};

% fid = fopen('lossData1.txt');
% data = textscan(fid, '%d %f', 'CommentStyle','#', 'CollectOutput',true);
% fclose(fid);
% lossData_itr1 = data{1,1};
% lossData_err1 = data{1,2};

clear data numbers a

%% Plot the error curve
figure
semilogy(lossData_itr,lossData_err,'b-','LineWidth',1.5)
ylabel('$\log_{10}$(error)','Interpreter','latex')
xlabel('Iterations','Interpreter','latex')
ylim([0.005,1])
% legend({'No pre-training'},'Interpreter','latex')
set(gca,'ticklabelinterpreter','latex','fontsize',16)
hold on
% semilogy(lossData_itr1,lossData_err1,'r-','LineWidth',1.5)
% legend({'No pretraining','Pretraining'},'Interpreter','latex')

%% Now we can start plotting figures.
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
zMin=-1;
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
    set(gca,'ticklabelinterpreter','latex','fontsize',21)
    colormap(jet);
    colorbar('ticklabelinterpreter','latex')
end