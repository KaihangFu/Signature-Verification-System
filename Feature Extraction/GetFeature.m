function feature = GetFeature(imagename)
%Function GetFeature

%Signature Preprocessing
I=imread(imagename);
I=rgb2gray(I);
I=im2double(I);
I=imadjust(I);
I=im2bw(I);
%Crop the signature image
[m,n]=size(I);
xstart=n;
xend=1;
ystart=m;
yend=1;
for r=1:m
    for c=1:n
        if (I(r,c)==0)
            if (r<ystart)
                ystart=r;
            end
            if (r>yend)
                yend=r; 
            end
            if (c<xstart)
                xstart=c;
            end
            if (c>xend)
                xend=c;
            end     
        end  
    end
end
for i=ystart:yend
    for j=xstart:xend
        I1((i-ystart+1),(j-xstart+1))=I(i,j);
    end
end

I2=imresize(I1,[256,256]);
I3=bwmorph(~I2, 'thin', inf);
I3=~I3;

%Signature Feature Extraction

%Global Feature

%number of signature pixels(in thinned image)
k=1;
for i1=1:256
    for j1=1:256
	    if(I3(i1,j1)==0)
		   k=k+1;
		end
	end
end
N = k-1;   %the total number of pixels in the thinned signature

%area of the signature(in binary image)
a=1;
for i5=1:256
    for j5=1:256
	    if(I2(i5,j5)==0)
		   a=a+1;
		end
	end
end
area = a-1;   %the total number of pixels in the signature

%entropy(in grayscale image)
%Entropy(i,1) = entropy(I);

%aspect ratio(in binary image)
[m,n]=size(I1);
xstart=n;
xend=1;
ystart=m;
yend=1;
for r=1:m
    for c=1:n
        if (I1(r,c)==0)
            if (r<ystart)
                ystart=r;
            end
            if (r>yend)
                yend=r; 
            end
            if (c<xstart)
                xstart=c;
            end
            if (c>xend)
                xend=c;
            end     
       end  
    end
end
Width = xend-xstart;
Height = yend-ystart;
aspect_ratio = Width/Height;

%The inclination angle between signature center of gravity and the lower right corner(in binary image)
%vertical variance & horizontal variance(in binary image)
[m,n]=size(I1);
t=1;
for i2=1:m
    for j2=1:n
	    if(I1(i2,j2)==0)
		   u(t)=i2;
		   v(t)=j2;
		   t=t+1;
		end
	end
end
C = [u;v];   %the curve of the signature
T = t-1;   %the total number of pixels in the signature
Xg = sum(C(2,:))/T;   %the x co-ordinate center of gravity of the image
Yg = sum(C(1,:))/T;   %the y co-ordinate center of gravity of the image
Angle = atan(abs(m-Yg)/abs(n-Xg));   %the inclination angle between signature center of gravity and the lower right corner 
H_var = sum(((C(2,:)-Xg)/n).^2);   %horizontal variance of signature 
V_var = sum(((C(1,:)-Yg)/m).^2);   %vertical variance of signature 
clear u v;

%intersection points & border points(in thinned signature) 
w=0;
ip_number=1;
bp_number=1;
for i3=2:255
    for j3=2:255
	    if(I3(i3,j3)==0)
	        if(I3(i3-1,j3-1)==0)
		       w=w+1;
		    end
		    if(I3(i3-1,j3)==0)
		       w=w+1;
		    end
		    if(I3(i3-1,j3+1)==0)
		       w=w+1;
		    end
		    if(I3(i3,j3-1)==0)
		       w=w+1;
		    end
		    if(I3(i3,j3+1)==0)
		       w=w+1;
		    end
		    if(I3(i3+1,j3-1)==0)
		       w=w+1;
		    end
		    if(I3(i3+1,j3)==0)
		       w=w+1;
		    end
		    if(I3(i3+1,j3+1)==0)
		       w=w+1;
		    end
		
		    if(w==1)
		       bp_number = bp_number+1;
		    end
		    if(w>=3)
		       ip_number = ip_number+1;
		    end
		end
		w=0;
	end
end	
BP_number = bp_number-1;
IP_number = ip_number-1;

%Local Feature

%signature density around the center of gravity
%the average angle of inclination of each black pixel in each cell to the lower right corner of the cell
%the distance between each black pixel in each cell with the lower right corner of the cell
[m,n]=size(I1);

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=1:fix(Yg/2)
    for j4=1:fix(Xg/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs((Yg/2)-i4)/abs((Xg/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt(((Yg/2)-i4)^2+((Xg/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   %the total number of pixels in this part signature
R1 = S/(fix(Xg/2)*fix(Yg/2));   %the density in this part signature
if(S==0)
   PA1 = 0;   
   PD1 = 0;
else 
   PA1 = Angles_sum/S;    %the average angle in this part signature
   PD1 = Distances_sum/S;    %the average distance in this part signature
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=1:fix(Yg/2)
    for j4=fix(Xg/2):fix(Xg)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs((Yg/2)-i4)/abs(Xg-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt(((Yg/2)-i4)^2+(Xg-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R2 = S/(fix(Xg/2)*fix(Yg/2));   
if(S==0)
   PA2 = 0;   
   PD2 = 0;
else 
   PA2 = Angles_sum/S;    
   PD2 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg/2):fix(Yg)
    for j4=1:fix(Xg/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(Yg-i4)/abs((Xg/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((Yg-i4)^2+((Xg/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R3 = S/(fix(Xg/2)*fix(Yg/2)); 
if(S==0)
   PA3 = 0;   
   PD3 = 0;
else 
   PA3 = Angles_sum/S;    
   PD3 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg/2):fix(Yg)
    for j4=fix(Xg/2):fix(Xg)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(Yg-i4)/abs(Xg-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((Yg-i4)^2+(Xg-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R4 = S/(fix(Xg/2)*fix(Yg/2)); 
if(S==0)
   PA4 = 0;   
   PD4 = 0;
else 
   PA4 = Angles_sum/S;    
   PD4 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=1:fix(Yg/2)
    for j4=fix(Xg):fix((n+Xg)/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs((Yg/2)-i4)/abs(((n+Xg)/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt(((Yg/2)-i4)^2+(((n+Xg)/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R5 = S/(fix((n-Xg)/2)*fix(Yg/2));
if(S==0)
   PA5 = 0;   
   PD5 = 0;
else 
   PA5 = Angles_sum/S;    
   PD5 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=1:fix(Yg/2)
    for j4=fix((n+Xg)/2):n
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs((Yg/2)-i4)/abs(n-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt(((Yg/2)-i4)^2+(n-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R6 = S/(fix((n-Xg)/2)*fix(Yg/2));
if(S==0)
   PA6 = 0;   
   PD6 = 0;
else 
   PA6 = Angles_sum/S;    
   PD6 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg/2):fix(Yg)
    for j4=fix(Xg):fix((n+Xg)/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(Yg-i4)/abs(((n+Xg)/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((Yg-i4)^2+(((n+Xg)/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R7 = S/(fix((n-Xg)/2)*fix(Yg/2));
if(S==0)
   PA7 = 0;   
   PD7 = 0;
else 
   PA7 = Angles_sum/S;    
   PD7 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg/2):fix(Yg)
    for j4=fix((n+Xg)/2):n
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(Yg-i4)/abs(n-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((Yg-i4)^2+(n-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R8 = S/(fix((n-Xg)/2)*fix(Yg/2));
if(S==0)
   PA8 = 0;   
   PD8 = 0;
else 
   PA8 = Angles_sum/S;    
   PD8 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg):fix((m+Yg)/2)
    for j4=1:fix(Xg/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(((m+Yg)/2)-i4)/abs((Xg/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((((m+Yg)/2)-i4)^2+((Xg/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R9 = S/(fix(Xg/2)*fix((m-Yg)/2));
if(S==0)
   PA9 = 0;   
   PD9 = 0;
else 
   PA9 = Angles_sum/S;    
   PD9 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg):fix((m+Yg)/2)
    for j4=fix(Xg/2):fix(Xg)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(((m+Yg)/2)-i4)/abs(Xg-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((((m+Yg)/2)-i4)^2+(Xg-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R10 = S/(fix(Xg/2)*fix((m-Yg)/2));
if(S==0)
   PA10 = 0;   
   PD10 = 0;
else 
   PA10 = Angles_sum/S;    
   PD10 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix((m+Yg)/2):m
    for j4=1:fix(Xg/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(m-i4)/abs((Xg/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((m-i4)^2+((Xg/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R11 = S/(fix(Xg/2)*fix((m-Yg)/2));
if(S==0)
   PA11 = 0;   
   PD11 = 0;
else 
   PA11 = Angles_sum/S;    
   PD11 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix((m+Yg)/2):m
    for j4=fix(Xg/2):fix(Xg)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(m-i4)/abs(Xg-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((m-i4)^2+(Xg-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R12 = S/(fix(Xg/2)*fix((m-Yg)/2));
if(S==0)
   PA12 = 0;   
   PD12 = 0;
else 
   PA12 = Angles_sum/S;    
   PD12 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg):fix((m+Yg)/2)
    for j4=fix(Xg):fix((n+Xg)/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(((m+Yg)/2)-i4)/abs(((n+Xg)/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((((m+Yg)/2)-i4)^2+(((n+Xg)/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R13 = S/(fix((n-Xg)/2)*fix((m-Yg)/2));
if(S==0)
   PA13 = 0;   
   PD13 = 0;
else 
   PA13 = Angles_sum/S;    
   PD13 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix(Yg):fix((m+Yg)/2)
    for j4=fix((n+Xg)/2):n
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(((m+Yg)/2)-i4)/abs(n-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((((m+Yg)/2)-i4)^2+(n-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R14 = S/(fix((n-Xg)/2)*fix((m-Yg)/2));
if(S==0)
   PA14 = 0;   
   PD14 = 0;
else 
   PA14 = Angles_sum/S;    
   PD14 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix((m+Yg)/2):m
    for j4=fix(Xg):fix((n+Xg)/2)
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(m-i4)/abs(((n+Xg)/2)-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((m-i4)^2+(((n+Xg)/2)-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R15 = S/(fix((n-Xg)/2)*fix((m-Yg)/2));
if(S==0)
   PA15 = 0;   
   PD15 = 0;
else 
   PA15 = Angles_sum/S;    
   PD15 = Distances_sum/S;
end

K=1;
Angles_sum=0;
Distances_sum=0;
for i4=fix((m+Yg)/2):m
    for j4=fix((n+Xg)/2):n
	    if(I1(i4,j4)==0)
		   K=K+1;
		   pixel_angle = atan(abs(m-i4)/abs(n-j4));
		   Angles_sum = Angles_sum + pixel_angle;
		   pixel_distance = sqrt((m-i4)^2+(n-j4)^2);
		   Distances_sum = Distances_sum + pixel_distance;
		end
	end
end
S = K-1;   
R16 = S/(fix((n-Xg)/2)*fix((m-Yg)/2));
if(S==0)
   PA16 = 0;   
   PD16 = 0;
else 
   PA16 = Angles_sum/S;    
   PD16 = Distances_sum/S;
end

feature = [N, area, aspect_ratio, Angle, H_var, V_var, BP_number, IP_number, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, PA1, PA2, PA3, PA4, PA5, PA6, PA7, PA8, PA9, PA10, PA11, PA12, PA13, PA14, PA15, PA16, PD1, PD2, PD3, PD4, PD5, PD6, PD7, PD8, PD9, PD10, PD11, PD12, PD13, PD14, PD15, PD16];

end
