library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity qMultiplier is
  	generic (
  			m            : Integer :=1;
  			n            : Integer :=7
  	);
  	port    (
  			A,B: 		in std_logic_vector((n+m+1)downto 1);
  			clk:		in std_logic;
  			en:			in std_logic;
           	output: 	out std_logic_vector((n+m+1)downto 1)
    );
end entity qMultiplier;

architecture behavioral of qMultiplier is
	
signal soutput : std_logic_vector((n+m+1)downto 1);

begin
	process(clk)
		
	variable extendA, extendB: std_logic_vector((n+m+1)*2 downto 1) := (others => '0');
	variable prodResult: std_logic_vector((n+m+1)*4 downto 1) := (others => '0');
	variable actualResult: std_logic_vector((n+m+1)*2 downto 1);
		
	begin
		if(clk'event and clk = '1' and en = '1') then
			-- Perform sign extend
	    	extendA := std_logic_vector(resize(signed(A), extendA'length));
	    	extendB := std_logic_vector(resize(signed(B), extendB'length));
	    	
	    	-- Do multiplication
	    	prodResult := std_logic_vector(unsigned(extendA) * unsigned(extendB));
	    	
	    	-- Only the least significant values are correct
	    	actualResult := prodResult((n+m+1)*2 downto 1);
	    	
	    	-- Truncate result
	    	soutput <= actualResult((2*n+m+1) downto n+1);
		end if;
  	end process;
	output <= soutput;
end architecture;