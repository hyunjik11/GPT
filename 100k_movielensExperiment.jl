@everywhere using DataFrames
@everywhere using GPT_SGLD
@everywhere using Distributions

### data processing
@everywhere function getdummy{R}(df::DataFrame, cname::Symbol, ::Type{R})
    darr = df[cname]
    vals = sort(levels(darr))#[2:end]
    namedict = Dict(vals, 1:length(vals))   
    arr = zeros(R, length(darr), length(namedict))
    for i=1:length(darr)
        if haskey(namedict, darr[i])
            arr[i, namedict[darr[i]]] = 1
        end        
    end
    newdf = convert(DataFrame, arr)
    names!(newdf, [symbol("$(cname)_$k") for k in vals])
    return newdf
end

@everywhere function convertdummy{R}(df::DataFrame, cnames::Array{Symbol}, ::Type{R})
    # consider every variable from cnames as categorical
    # and convert them into set of dummy variables,
    # return new dataframe
    newdf = DataFrame()
    for cname in names(df)
        if !in(cname, cnames)
            newdf[cname] = df[cname]
        else
            dummydf = getdummy(df, cname, R)
            for dummyname in names(dummydf)
                newdf[dummyname] = dummydf[dummyname]
            end
        end
    end
    return newdf
end

@everywhere convertdummy(df::DataFrame, cnames::Array{Symbol}) = convertdummy(df, cnames, Int32)

@everywhere function bin_age(age::Array)
	q=quantile(age,[0.2,0.4,0.6,0.8,1.0])
	indmin(q.<UserData[30,2])
        map(x->indmin(q.<x),age)
end

##data clearing
@everywhere UserData = readdlm("ml-100k/u.user", '|');
@everywhere MovieData = readdlm("ml-100k/u.item",'|');
@everywhere Rating = readdlm("ml-100k/u.data",Float64);

@everywhere Ntrain = 80000;
@everywhere Ntest = 20000;
@everywhere UserData[:,2] = bin_age(UserData[:,2])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,3:end]; 
Nu,Du=size(UserData); Nm,Dm=size(MovieData); 
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere Ratingtrain=hcat(Rating[1:Ntrain,1:2],ytrain);
@everywhere Ratingtest=hcat(Rating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
@everywhere n = 10; 
@everywhere M = 5;
@everywhere burnin=0;
@everywhere numiter=30;
@everywhere r = 8
@everywhere Q=r;   
@everywhere D = 2;
@everywhere sigma = 0.3;
@everywhere signal_var = sigma^2;
@everywhere param_seed=17;
@everywhere I=repmat(1:r,1,2);
@everywhere m = 100;
@everywhere maxepoch = 20;

@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere a=1;b1=1;b2=2;
UserHashmap=Array(Int64,M,Nu); MovieHashmap=Array(Int64,M,Nm);
for i=1:Nu
	UserHashmap[:,i]=sample(1:n,M,replace=false)
end
for i=1:Nm
	MovieHashmap[:,i]=sample(1:n,M,replace=false)
end
UserBHashmap=2*rand(Bernoulli(),M,Nu)-1
MovieBHashmap=2*rand(Bernoulli(),M,Nm)-1
a=1;b1=1;b2=1;

# UserHashmap is an M by Nu array with M distinct hash values in 1:n for each col: sample(1:n,M,replace=false) Similar for MovieHashmap
# UserBHashmap is an M by Nu array of +/-1: 2*rand(Bernoulli(),M,Nu)-1 Similar for MovieBHashmap.
function CFfeature(UserData::Array,MovieData::Array,UserHashmap::Array,MovieHashmap::Array, UserBHashmap::Array,MovieBHashmap::Array,n::Integer,a::Real,b1::Real,b2::Real)
	Nu,Du=size(UserData); Nm,Dm=size(MovieData); # number of users/movies and features
	phiUser=spzeros(n+Du,Nu);
	phiMovie=spzeros(n+Dm,Nm);
	for user=1:Nu
		for j=1:M
			phiUser[UserHashmap[j,user],user]=UserBHashmap[j,user];
		end
	end
	phiUser[1:n,:]*=a/M			
	phiUser[n+1:n+Du,:]=b1*UserData'
	for movie=1:Nm
		for j=1:M
			phiMovie[MovieHashmap[j,movie],movie]=MovieBHashmap[j,movie];
		end
	end
	phiMovie[1:n,:]*=1/M		
	phiMovie[n+1:n+Dm,:]=b2*MovieData'
	return phiUser,phiMovie
end

# to get the feature for user u,movie m, take the kronecker product kron(phiUser[:,u],phiMovie[:,m])
function CFfeatureNotensor(myRating::Array,UserData::Array,MovieData::Array,UserHashmap::Array,MovieHashmap::Array, UserBHashmap::Array,MovieBHashmap::Array,n::Integer,a::Real,b1::Real,b2::Real)
    Rating=int(myRating[:,1:2]);
        N=size(Rating,1);
	Nu,Du=size(UserData); Nm,Dm=size(MovieData);
	phiUser,phiMovie=CFfeature(UserData,MovieData,UserHashmap,MovieHashmap,UserBHashmap,MovieBHashmap, n,a,b1,b2);
	phi=zeros((n+Du)*(n+Dm),N);
	for i=1:N
		phi[:,i]=kron(phiUser[:,Rating[i,1]],phiMovie[:,Rating[i,2]])
	end
	return phi
end

function CFgradfeatureNotensor(myRating::Array,UserData::Array,MovieData::Array,UserHashmap::Array,MovieHashmap::Array, UserBHashmap::Array,MovieBHashmap::Array,n::Integer,a::Real,b1::Real,b2::Real)
    Rating=int(myRating[:,1:2]);
    N=size(Rating,1);
    Nu,Du=size(UserData); Nm,Dm=size(MovieData);
    Rating
    phiUser,phiMovie=CFfeature(UserData,MovieData,UserHashmap,MovieHashmap,UserBHashmap,MovieBHashmap, n,a,b1,b2);
    tempa=zeros(n+Du,Nu);tempa[1:n,:]=phiUser[1:n,:]/a;
    tempb1=zeros(n+Du,Nu);tempb1[n+1:n+Du,:]=phiUser[n+1:n+Du,:]/b1;
    tempb2=zeros(n+Dm,Nm);tempb2[n+1:n+Dm,:]=phiMovie[n+1:n+Dm,:]/b2;

    gradphia=zeros((n+Du)*(n+Dm),N);
    gradphib1=zeros((n+Du)*(n+Dm),N);
    gradphib2=zeros((n+Du)*(n+Dm),N);
    for i=1:N
        gradphia[:,i]=kron(tempa[:,Rating[i,1]],phiMovie[:,Rating[i,2]])
        gradphib1[:,i]=kron(tempb1[:,Rating[i,1]],phiMovie[:,Rating[i,2]])
        gradphib2[:,i]=kron(phiUser[:,Rating[i,1]],tempb2[:,Rating[i,2]])
    end
    return cat(3,gradphia,gradphib1,gradphib2)
end

@time Z1=CFfeature(UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);
@time Z2=CFfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);
@time Z3=CFgradfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);


