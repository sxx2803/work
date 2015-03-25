library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron is
	generic (
		numInputs: integer := 3;
		n: integer := 7;
		m: integer := 1
	);
	port (
		clk : in std_logic;
		en:		in std_logic;
		dataIn: in std_logic_vector(numInputs*(n+m+1) downto 1);
		dataOut: out std_logic_vector(n+m+1 downto 1)
	);
end entity neuron;

architecture behavioral of neuron is
	
component sigmoid
	generic (
		n: integer := 1;
		m: integer := 7);
	port (
		sigmoidIn: in std_logic_vector((n+m+1) downto 1);
		sigmoidOut: out std_logic_vector((n+m+1) downto 1)
	);
end component;
	
signal ssum: std_logic_vector(n+m+1 downto 1);
	
begin
	
	sigmoidBlock: sigmoid 	generic map(m=>m, n=>n)
							port map(sigmoidIn=>ssum, sigmoidOut=>dataOut);
	
	addProcess: process(clk)
		variable tempSum: std_logic_vector(n+m+1 downto 1) := (others => '0');
		variable chkOverflow: std_logic_vector(n+m+1 downto 1) := (others => '0');
		variable indexHi: integer;
		variable indexLo: integer;
	begin
		if(clk = '1' and clk'event and en = '1') then
		  	-- Reset tempSum
		  	tempSum := (others => '0');
		  	chkOverflow := (others => '0');
		  	for ii in 1 to numInputs loop
				indexHi := ii*(n+m+1);
				indexLo := ((ii-1)*(n+m+1))+1;
				if((tempSum(n+m+1) xnor dataIn(indexHi)) = '0') then
					--report "Signs are different for MSBs: " & std_logic'image(tempSum(n+m+1)) & " and " & std_logic'image(dataIn(indexHi));
					-- If signs are different, can never overflow
					tempSum := std_logic_vector(signed(tempSum) + signed(dataIn(indexHi downto indexLo)));
				else
					--report "Signs are same for MSBs: " & std_logic'image(tempSum(n+m+1)) & " and " & std_logic'image(dataIn(indexHi));
					-- Signs are same, potential for overflow
					chkOverflow := std_logic_vector(signed(tempSum) + signed(dataIn(indexHi downto indexLo)));
					-- Check if output MSB is different than input MSB
					if((chkOverflow(n+m+1) xnor dataIn(indexHi)) = '0') then
						report "Overflow detected, setting to max";
						-- Check if to set overflow to most negative or most positive
						if(dataIn(indexHi) = '1') then
							tempSum := (others => '0');
						else
							tempSum := (others => '1');
						end if;
						-- Keep signed bit
						tempSum(n+m+1) := dataIn(indexHi);
					else
						tempSum := std_logic_vector(signed(tempSum) + signed(dataIn(indexHi downto indexLo)));
					end if;
				end if;
			end loop;
			ssum <= tempSum;
		end if;
	end process;
	
end architecture behavioral;