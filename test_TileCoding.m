function test_TileCoding

tileC = TileCoding(1, 100,0,100,1);

for x =1:100
   assert(x==c2d(tileC, x)); 
end 


tileC = TileCoding(2, [100 100],[0 0],[100 100],1);

for x =1:100
    d0 = c2d(tileC, [x 0]);
    assert(d0== 100*x);
    for y =1:100
        dy = c2d(tileC, [x y]);
        assert(dy-d0==y);
    end
end 


tileC = TileCoding(1, 100,0,100,2);

for x =2:100
   xd = c2d(tileC, x);
   assert(x==xd(1) && x-1==xd(2))
end 

for x = 1.7:100
   xd = c2d(tileC, x);
   assert(floor(x)==xd(1) && floor(x)==xd(2))
end




end