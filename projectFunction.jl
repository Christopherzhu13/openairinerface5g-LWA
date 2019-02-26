using LibPQ, DataStreams, DataFrames
function getProjectData()
connection_dir = "./postgresql-files/missing-pitch"

# enter your group's username and password as strings here:
username = "user1"
password = "75cd34ab283d2142fde3662d567b15db3f4c706391ac8445796ddeee24fcbcdc"

conn = connect_missing_pitch(connection_dir, username, password)
    return conn
end

function pitchpype2num(data)
data[data[:,1].=="CH",1].="1"
data[data[:,1].=="CU",1].="2"
data[data[:,1].=="FC",1].="3"
data[data[:,1].=="FF",1].="4"
data[data[:,1].=="FO",1].="5"
data[data[:,1].=="FS",1].="6"
data[data[:,1].=="FT",1].="7"
data[data[:,1].=="KC",1].="8"
data[data[:,1].=="KN",1].="9"
data[data[:,1].=="SC",1].="10"
data[data[:,1].=="S2",1].="11"
data[data[:,1].=="SL",1].="12"
#data_Array=convert(Array, data)
data_Array[:,1]=map(x->parse(Float64,x),data_Array[:,1]) 
    return data_Array
end

function num2pitchtype(data)
data_Array=convert(Array, data)
data_Array[:,1]=map(x->parse(Float64,x),data_Array[:,1]) 
data[data[:,1].=="CH",1]="1"
data[data[:,1].=="CU",1]="2"
data[data[:,1].=="FC",1]="3"
data[data[:,1].=="FF",1]="4"
data[data[:,1].=="FO",1]="5"
data[data[:,1].=="FS",1]="6"
data[data[:,1].=="FT",1]="7"
data[data[:,1].=="KC",1]="8"
data[data[:,1].=="KN",1]="9"
data[data[:,1].=="SC",1]="10"
data[data[:,1].=="S2",1]="11"
data[data[:,1].=="SL",1]="12"
data_Array=convert(Array, data)
data_Array[:,1]=map(x->parse(Float64,x),data_Array[:,1]) 
    return data_Array
end

function completeAsvd(A,k,iters)
    error_tol=1e-9;
    MissingEntries = ismissing.(A)
    NotMissingEntries = .!MissingEntries
    Learnerror = 1000000000
    Learnerr = []
    idx = 1 # Iteration index
    
    A_hat = collect(Missings.replace(A, 0.0)) # Estimate
    A_hat=convert(Array{Float64,2},A_hat)
    while (Learnerror > error_tol && idx <= iters)
        
        A_hat[NotMissingEntries] = A[NotMissingEntries] # Force Known Entries: Projection Step
                
        U, s, V = svd(A_hat)
        A_hat=U[:,1:k]*Diagonal(s[1:k])*V[:,1:k]'
        
        Learnerror = sqrt(sum(abs2, skipmissing(A - A_hat)) / sum(abs2, skipmissing(A))) # Normalized error on known entries
        push!(Learnerr,Learnerror)
        
        idx = idx + 1
        
    end
 
    return A_hat, Learnerr
end



function optshrink2(Y, r)
    # U is [m, min(m,n)], s is min(m,n), V is [n, min(n,m)]
    (U, s, V) = svd(Y)

    (m, n) = size(Y)
    r = minimum([r, m, n]) # ensure r <= min(m,n)

    sv = s[r+1:end] # tail singular values for noise estimation
    D_sum=0;
    w_sum=0;
    w = zeros(r)
    for k=1:r
        (D, Dder) = D_transform_from_vector(s[k], sv, m, n)
        w[k] = -2*D/Dder
        D_sum=D_sum+1/D
        w_sum=w_sum+w[k]^2;
    end
   
    Xh = U[:,1:r]*Diagonal(w)*V[:,1:r]'
    MSE=1-w_sum/D_sum
    return Xh,MSE
end

function D_transform_from_vector(z, sn, m, n)
    # sn is of length n <= m

    sm = [sn; zeros(m - n)] # m x 1

    inv_n = 1 ./ (z^2 .- sn.^2) # vector corresponding to diagonal
    inv_m = 1 ./ (z^2 .- sm.^2)

    D1 = (1/n)*sum(z*inv_n)
    D2 = (1/m)*sum(z*inv_m)

    D = D1*D2 # eq (16a) in  paper

    # derivative of D transform
    D1_der = (1/n)*sum(-2*z^2 .* inv_n.^2 + inv_n)
    D2_der = (1/m)*sum(-2*z^2 .* inv_m.^2 + inv_m)

    D_der = D1*D2_der + D2*D1_der # eq (16b) in  paper

    return (D, D_der)
end