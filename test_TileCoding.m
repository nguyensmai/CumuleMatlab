function test_TileCoding

%simple case
tileC = TileCoding(1, 100,1,100,1);

x= 1; d =1;   assert(d==c2d(tileC, x));
x= 2; d =2;   assert(d==c2d(tileC, x));
x= 99;d =99;  assert(d==c2d(tileC, x));
x= 100;d =100;assert(d==c2d(tileC, x));
x= 101;d =100;assert(d==c2d(tileC, x));

for x =1:100
   assert(x==c2d(tileC, x)); 
end 

%dimension = 2
dimTD     = 2;
nbLayers  = 1;
nbTiles   = 100;
tileMin   = ones(1,dimTD);   
tileMax   = 100*ones(1,dimTD);
tileC     = TileCoding(dimTD, nbTiles*ones(1,dimTD),tileMin,tileMax,nbLayers);

x= 1;  y =1;  d =1;   assert(d==c2d(tileC, [x y])); 
x= 1;  y =2;  d =2;   assert(d==c2d(tileC, [x y])); 
x= 1;  y =99; d =99; assert(d==c2d(tileC, [x y])); 
x= 1;  y =100;d =100; assert(d==c2d(tileC, [x y])); 
x= 2;  y =1;  d =101; assert(d==c2d(tileC, [x y])); 
x= 2;  y =99;  d =199;assert(d==c2d(tileC, [x y])); 
x= 2;  y =100; d =200; assert(d==c2d(tileC, [x y])); 
x= 100; y =100;  d =10^4;assert(d==c2d(tileC, [x y])); 

for x =1:100
    d0 = c2d(tileC, [x 1]);
    assert(d0(1) == nbTiles*(x-1)+1);
    for y =2:100
        dy = c2d(tileC, [x y]);
        assert(dy-d0==y-1);
    end
end 

%2 layers
tileC = TileCoding(1, 100,1,100,2);

xd = c2d(tileC, 1);
assert(1==xd(1) && 1==xd(2))

for x =2:100
   xd = c2d(tileC, x);
   assert(x==xd(1) && x-1==xd(2))
end 

xd = c2d(tileC, 1.2);
assert(1==xd(1) && 1==xd(2))
for x = 2.2:100
   xd = c2d(tileC, x);
   assert(floor(x)==xd(1) && floor(x)-1==xd(2))
end

xd = c2d(tileC, 1.7);
assert(1==xd(1) && 1==xd(2))
for x = 2.7:100
   xd = c2d(tileC, x);
   assert(floor(x)==xd(1) && floor(x)==xd(2))
end


%dimension = 2 , 2 layers
dimTD     = 2;
nbLayers  = 2;
nbTiles   = 100;
tileMin   = ones(1,dimTD);   
tileMax   = 100*ones(1,dimTD);
tileC     = TileCoding(dimTD, nbTiles*ones(1,dimTD),tileMin,tileMax,nbLayers);

x= 1;  y =1;  d =[1;1];   assert(all(d==c2d(tileC, [x y]))); 
x= 1;  y =2;  d =[2;1];   assert(all(d==c2d(tileC, [x y]))); 
x= 1;  y =3;  d =[3;2];   assert(all(d==c2d(tileC, [x y])));  
x= 1;  y =99; d =[99;98]; assert(all(d==c2d(tileC, [x y])));  
x= 1;  y =100;d =[100;99];assert(all(d==c2d(tileC, [x y])));  
x= 2;  y =1;  d =[101;1]; assert(all(d==c2d(tileC, [x y])));  
x= 2;  y =2;  d =[102;1]; assert(all(d==c2d(tileC, [x y])));  
x= 2;  y =100;d =[200;99];assert(all(d==c2d(tileC, [x y])));  
x= 100;  y =100;d =[100^2;9899];assert(all(d==c2d(tileC, [x y])));  


end