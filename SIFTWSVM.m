function testY=SIFTWSVM(testX,A,B,C,c1,c2,c3,c4,alpha)
%A:+1，B:-1，C: no label
%q:absolute value of positive and negative sample difference
%beta:weight bounds，
%alpha: gap parameter
%c1,c2,c3,c4 model parameter
%一、main program
n1=size(A,1);
n2=size(B,1);
q=abs(n1-n2);
% initialization
[u1,u2]=TSVM(A,B,c1,c2,c3,c4);%u1=[w1;b1]
[Yc,C,D]=GY4(A,B,C,u1,u2,c1,c2,c3,c4,n1,n2,q);%给Y贴标签
m=ceil(0.1*size(C,1));
while(1)
    [A,B,C,D,Yc,u1,u2,sign]=GZ(A,B,C,D,Yc,u1,u2,alpha,c1,c2,c3,c4,n1,n2,q,m);
    if sign==0
        sprintf('.');
        break;
    end
    a=size(C,1);
    if a==0
        break;
    end
end
[~,n]=size(A);
w11=u1(1:n,1);
b11=u1(n+1,1);
w22=u2(1:n,1);
b22=u2(n+1,1);
m=size(testX,1);
testY=ones(m,1);
for i=1:m
    x=testX(i,:);
    dis1=abs(x*w11+b11)/norm(w11);
    dis2=abs(x*w22+b22)/norm(w22);
    if dis1>dis2
        testY(i,1)=-1;
    end
end
end
%二、four subprograms
%函数GZ
function [A,B,C,D,Yc,u1,u2,sign]=GZ(A,B,C,D,Yc,u1,u2,alpha,c1,c2,c3,c4,n1,n2,q,m)
sign=1;
%the first part is to find the weight of A, B, C
[sA,sB,sC]=GSabc1(A,B,C,Yc,u1,u2,alpha);
if sum(sA)<1e-10 || sum(sB)<1e-10
    sign=0;
    return;
end
%The second part, supplement the point of m in the weight in C to A or B
[A,B,C,sA,sB,uns]=GAB2(A,B,C,D,sA,sB,sC,Yc,m);
if  uns~=0
    sign=0;
    return;
end
%The third part, update u1, u2
[u1,u2]=GU3(A,B,sA,sB,c1,c2,c3,c4);
if sum(abs(u1))==0 || sum(abs(u2))==0
    sign=0;
    return;
end
%Part four, label C
if size(C,1)~=0
    [Yc,C,D]=GY4(A,B,C,u1,u2,c1,c2,c3,c4,n1,n2,q);
end
end

%函数GSabc1
function [sA,sB,sC]=GSabc1(A,B,C,Yc,u1,u2,alpha)
delta=10e-6;
[m1,n]=size(A);
m2=size(B,1);
m3=size(C,1);
w1=u1(1:n,:);
b1=u1(n+1,:);
w2=u2(1:n,:);
b2=u2(n+1,:);
LA1=zeros(m1,1);%A中正类点到正类线的距离（带符号）
LA2=zeros(m2,1);%B中负类点到正类线的距离（带符号）
LB1=zeros(m2,1);%B中负类点到负类线的距离（带符号）
LB2=zeros(m1,1);%B中正类点到负类线的距离（带符号）
LC1=zeros(m3,1);%无标签点到正类线的距离（带符号）
LC2=zeros(m3,1);%无标签点到负类线的距离（带符号）
for i=1:m1
    LA1(i,1)=(A(i,:)*w1+b1)/norm(w1);
end
for i=1:m2
    LA2(i,1)=(B(i,:)*w1+b1)/norm(w1);
end
for i=1:m2
    LB1(i,1)=(B(i,:)*w2+b2)/norm(w2);
end
for i=1:m1
    LB2(i,1)=(A(i,:)*w2+b2)/norm(w2);
end
for i=1:m3
    LC1(i,1)=(C(i,:)*w1+b1)/norm(w1);
end
for i=1:m3
    LC2(i,1)=(C(i,:)*w2+b2)/norm(w2);
end
%求rA,正类点的最大距离
%求rB,负类点的最大距离
rA=max(abs(LA1));
rB=max(abs(LB1));
%求deltaA,deltaB
if size(LC1(Yc==1,1),1)~=0
    rC1=abs(max(abs(LC1(Yc==1,:))));
    if rA<rC1
        rA=rC1;
    end
end
if size(LC1(Yc==-1,1),1)~=0
    rC2=abs(max(abs(LC1(Yc==-1,:))));
    if rB<rC2
        rB=rC2;
    end
end
% 求muA,muB,muC
muA=zeros(m1,1);
for i=1:m1
    dis=abs( LA1(i,1));
    muA(i,1)=1-dis/(rA+delta);
end
muB=zeros(m2,1);
for i=1:m2
    dis=abs( LB1(i,1));
    muB(i,1)=1-dis/(rB+delta);
end
muC=zeros(m3,1);
for i=1:m3    
    if Yc(i,1)==1
        dis=abs( LC1(i,1));
        muC(i,1)=1-dis/(rA+delta);
    else
        dis=abs( LC2(i,1));
        muC(i,1)=1-dis/(rB+delta);
    end
end
%求rhoA，rhoB rhoC
rhoA=zeros(m1,1);
for i=1:m1
    numup=0;
    for j=1:m2
        tis= LA1(i,1)*LA2(j,1);
        if  tis>=0
            dis=abs(abs( LA1(i,1))-abs( LA2(j,1)));
        else
            dis=abs(abs( LA1(i,1))+abs( LA2(j,1)));
        end
        if dis<=alpha
            numup=numup+1;
        end
    end
    numdown=numup;
    for j=1:m1
        tis= LA1(i,1)*LA1(j,1);
        if  tis>=0
            dis=abs(abs( LA1(i,1))-abs( LA1(j,1)));
        else
            dis=abs(abs( LA1(i,1))+abs( LA1(j,1)));
        end
        if dis<=alpha
            numdown=numdown+1;
        end
    end
    rhoA(i,1)=numup/numdown;
end
rhoB=zeros(m2,1);
for i=1:m2
    numup=0;
    for j=1:m1
        tis= LB1(i,1)*LB2(j,1);
        if  tis>=0
            dis=abs(abs( LB1(i,1))-abs( LB2(j,1)));
        else
            dis=abs(abs( LB1(i,1))+abs( LB2(j,1)));
        end
        if dis<=alpha
            numup=numup+1;
        end
    end
    numdown=numup;
    for j=1:m2
        tis= LB1(i,1)*LB1(j,1);
        if  tis>=0
            dis=abs(abs( LB1(i,1))-abs( LB1(j,1)));
        else
             dis=abs(abs( LB1(i,1))+abs( LB1(j,1)));
        end
        if dis<=alpha
            numdown=numdown+1;
        end
    end
    rhoB(i,1)=numup/numdown;
end
rhoC=zeros(m3,1);
for i=1:m3
    numup=0;
    if Yc(i,1)==1
        for j=1:m2
            tis= LC1(i,1)*LA2(j,1);
            if tis>=0
                dis=abs(abs( LC1(i,1))-abs( LA2(j,1)));
            else
                dis=abs(abs( LC1(i,1))+abs( LA2(j,1)));
            end
            if dis<=alpha
                numup=numup+1;
            end
        end
        numdown=numup;
        for j=1:m1
            tis= LC1(i,1)*LA1(j,1);
            if tis>=0
                dis=abs(abs( LC1(i,1))-abs( LA1(j,1)));
            else 
                dis=abs(abs( LC1(i,1))+abs( LA1(j,1)));
            end
            if dis <=alpha
                numdown=numdown+1;
            end
        end
        rhoC(i,1)=numup/(numdown+1);
    else
        for j=1:m1
            tis= LC2(i,1)*LB2(j,1);
            if tis>=0
                dis=abs(abs( LC2(i,1))-abs( LB2(j,1)));
            else
                dis=abs(abs( LC2(i,1))+abs( LB2(j,1)));
            end
            if dis<=alpha
                numup=numup+1;
            end
        end
        numdown=numup;
        for j=1:m2
            tis= LC2(i,1)*LB1(j,1);
            if tis>=0
                dis=abs(abs( LC2(i,1))-abs( LB1(j,1)));
            else
                dis=abs(abs( LC2(i,1))+abs( LB1(j,1)));
            end
            if dis <=alpha
                numdown=numdown+1;
            end
        end
        rhoC(i,1)=numup/(numdown+1);
    end
end
%求nuA,nuB,nuC,
nuA=(1-muA).*rhoA;
nuB=(1-muB).*rhoB;
nuC=(1-muC).*rhoC;
%求sA,sB,sC
sA=zeros(m1,1);
for i=1:m1
    if nuA(i,1)<1e-10
        sA(i,1)=muA(i,1);
    elseif muA(i,1)<=nuA(i,1)
        sA(i,1)=0;
    elseif muA(i,1)>nuA(i,1)&& nuA(i,1)>1e-10
        sA(i,1)=(1-nuA(i,1))/(2-muA(i,1)-nuA(i,1));
    end
end
sB=zeros(m2,1);
for i=1:m2
    if nuB(i,1)<1e-10
        sB(i,1)=muB(i,1);
    elseif muB(i,1)<=nuB(i,1)
        sB(i,1)=0;
    elseif muB(i,1)>nuB(i,1)&& nuB(i,1)>1e-10
        sB(i,1)=(1-nuB(i,1))/(2-muB(i,1)-nuB(i,1));
    end
end
sC=zeros(m3,1);
for i=1:m3
    if nuC(i,1)<1e-10
        sC(i,1)=muC(i,1);
    elseif muC(i,1)<=nuC(i,1)
        sC(i,1)=0;
    elseif muC(i,1)>nuC(i,1)&& nuC(i,1)>1e-10
        sC(i,1)=(1-nuC(i,1))/(2-muC(i,1)-nuC(i,1));
    end
end
end

%function GAB2
function [A,B,C,sA,sB,uns]=GAB2(A,B,C,D,sA,sB,sC,Yc,m)
uns=0;
beta=0.1;
id=find(sC>=beta);
if isempty(id) 
    uns=size(C,1)+size(D,1);
    return
else
    C12=C(id,:);
    sC12=sC(id,:);
    Y12=Yc(id,:);
    n1=size(C12,1);
    [sC12,idex]=sort(sC12,'descend');
    C12=C12(idex,:);
    Y12=Y12(idex,:);
    C13=[];
    if m<n1
        C13=C12(m+1:end,:);
        n1=m;
    end
    for i=1:n1
        if Y12(i,1)==1
            A=[A;C12(i,:)];
            sA=[sA;sC12(i,1)];
        else
            B=[B;C12(i,:)];
            sB=[sB;sC12(i,1)];
        end
    end
    C(id,:)=[];
    C=[C;C13;D];
end
end

%function GU3
function [u1,u2]=GU3(A,B,sA,sB,c1,c2,c3,c4)
[m1,n]=size(A);
m2=size(B,1);
opt=optimoptions('quadprog','Display','off');
J=diag(sqrt(sA));
% compute u1
H1=[J*A J*ones(m1,1)];
H2=[B ones(m2,1)];
E=eye(n+1);
E(n+1,n+1)=0;
Temp=(H1'*H1*c1+E)\H2';
H=H2*Temp;
H=(H+H')/2;
e=-ones(m2,1);
alpha=quadprog(H,e,[],[],[],[],zeros(m2,1),c2*sB,[],opt);
if length(alpha)<=0
    u1=zeros(n+1,1);
    u2=zeros(n+1,1);
    return
end
u1=-Temp*alpha;

% compute u2
K=diag(sqrt(sB));
G1=[K*B K*ones(m2,1)];
G2=[A ones(m1,1)];
Temp=(G1'*G1*c3+E)\G2';
H=G2*Temp;
H=(H+H')/2;
e=-ones(m1,1);
alpha=quadprog(H,e,[],[],[],[],zeros(m1,1),c4*sA,[],opt);
if length(alpha)<=0
    u1=zeros(n+1,1);
    u2=zeros(n+1,1);
    return
end
u2=Temp*alpha;
end

%function GY4
function [Yc,C,D]=GY4(A,B,C,u1,u2,c1,c2,c3,c4,n1,n2,q)
D=[];
[m3,n]=size(C);
m1=size(A,1);
m2=size(B,1);
w1=u1(1:n,:);
b1=u1(n+1,:);
w2=u2(1:n,:);
b2=u2(n+1,:);
Yc=zeros(m3,1);
d1=zeros(m3,1);
d2=zeros(m3,1);
for i=1:m3
    d1(i,1)=(C(i,:)*w1+b1)^2/(2/c1)+c4*max(0,1-(C(i,:)*w2+b2));
    d2(i,1)=(C(i,:)*w2+b2)^2/(2/c3)+c2*max(0,1+(C(i,:)*w1+b1));
    if d1(i,1)>d2(i,1)
        Yc(i,1)=-1;
    else
        Yc(i,1)=1;
    end
end
d=sum(Yc)+(m1-n1)-(m2-n2);
if d>0&&abs(d)>q
    id=find(Yc==1);
    d11=d1(id,:);
    [~,idex]=sort(d11,'descend');
    id=id(idex,:);
    if abs(d)-q<size(id,1)
        id1=id(1:(abs(d)-q),1);
        Yc(id1,:)=[];
        D=C(id1,:);
        C(id1,:)=[];
    else
        Yc(id,:)=[];
        D=C(id,:);
        C(id,:)=[];
    end
elseif d<0&&abs(d)>q
    id=find(Yc==-1);
    d22=d2(id,1);
    [~,idex]=sort(d22,'descend');
    id=id(idex,:);
    if abs(d)-q<size(id,1)
        id1=id(1:(abs(d)-q),1);
        Yc(id1,:)=[];
        D=C(id1,:);
        C(id1,:)=[];
    else
        Yc(id,:)=[];
        D=C(id,:);
        C(id,:)=[];
    end
end
end

%TSVM
function [u1,u2]= TSVM(A,B,c1,c2,c3,c4)
opt=optimoptions('quadprog','Display','off');
% compute u1
[m1,n]=size(A);
m2=size(B,1);
H1=[A ones(m1,1)];
H2=[B ones(m2,1)];
E=eye(n+1);
E(n+1,n+1)=0;
Temp=(H1'*H1*c1+E)\H2';
H=H2*Temp;
H=(H+H')/2;
f=-ones(m2,1);
alpha=quadprog(H,f,[],[],[],[],zeros(m2,1),c2*ones(m2,1),[],opt);
u1=-Temp*alpha;

% compute u2
Temp=(H2'*H2*c3+E)\H1';
H=H1*Temp;
H=(H+H')/2;
f=-ones(m1,1);
alpha=quadprog(H,f,[],[],[],[],zeros(m1,1),c4*ones(m1,1),[],opt);
u2=Temp*alpha;
end