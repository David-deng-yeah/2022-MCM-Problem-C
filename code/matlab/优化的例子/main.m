
global i
resX = zeros(800, 3);
resY = zeros(800, 1);
for i=1:800
    [x,y]=fmincon('fun1',rand(3,1),[],[],[1 1 1],1,[0 0 0],[1 1 1]);
    resX(i, :) = x';
    resY(i, :) = y;
end