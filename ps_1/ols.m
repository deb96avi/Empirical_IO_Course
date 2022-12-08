function f = ols(b,y,X)
    f = 0;
    for k = 1:height(X)
        f = f + (y(k,1)-b(1)*X(k,1)-b(2)*X(k,2)-b(3)*X(k,3)-b(4)*X(k,4)-b(5)*X(k,5)-b(6))^2;
    end
end

