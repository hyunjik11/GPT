module testmod

export myfn,myfn2

function myfn(n)
	a=SharedArray(Float64,10)
	@sync @parallel for i=1:10
		println(i*n)
	end
	return sdata(a)
end

function myfn2(i)
	A=zeros(10)
	for k=1:i
		A+=myfn(1)
	end
	return A

end

end
