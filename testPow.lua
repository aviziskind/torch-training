--testPower

inf_val = 1/0
mod = nn.Sequential()
mod:add(nn.Power(inf_val))

im = torch.Tensor(5,5):zero()
for i = 1,5 do
    im[i] = i^2
end

print (mod:forward(im) )
