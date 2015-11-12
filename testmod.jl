module testmod

export myfn

function myfn()
	@parallel for i=1:10
		println(i)
	end
end

end
