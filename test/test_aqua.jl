@testitem "Aqua" tags = [:aqua] begin
    using NestedThreading, Aqua
    Aqua.test_all(NestedThreading)
end
