library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity node is
	generic (
		numInputs: integer := 3;
		n: integer := 7;
		m: integer := 1
	);
	port (
		clk : 		in std_logic;
		en:			in std_logic;
		weights:	in std_logic_vector(numInputs*(n+m+1) downto 1);
		xIn: 		in std_logic_vector(numInputs*(n+m+1) downto 1);
		yOut: 		out std_logic_vector(n+m+1 downto 1)
	);
end entity;

architecture behavioral of node is

component neuron
	generic (
		numInputs: integer := 3;
		n: integer := 7;
		m: integer := 1
	);
	port (
		clk: 	in std_logic;
		en:		in std_logic;
		dataIn: in std_logic_vector(numInputs*(n+m+1) downto 1);
		dataOut: out std_logic_vector(n+m+1 downto 1)
	);
end component;

component qMultiplier is
  	generic (m            : Integer :=1;
  			 n            : Integer :=7
  	);
  	port    (A,B          : in std_logic_vector((n+m+1)downto 1);
  			clk			  : in std_logic;
  			en			  : in std_logic;
           	output        : out std_logic_vector((n+m+1)downto 1)
    );
end component;
	
signal multiplyOut: std_logic_vector(numInputs*(n+m+1) downto 1);
	
begin
	
	Gen_QMultipliers:
	for ii in 1 to numInputs generate
		synapseX: qMultiplier 	generic map(m=>m, n=>n)
								port map( A=>xIn(ii*(n+m+1) downto ((ii-1)*(n+m+1))+1), 
									 	B=>weights(ii*(n+m+1) downto ((ii-1)*(n+m+1))+1), 
									 	output=>multiplyOut(ii*(n+m+1) downto ((ii-1)*(n+m+1))+1),
										en=>en,
										clk=>clk
										);
	end generate;
	
	NodeNeuron: neuron 	generic map(numInputs=>numInputs, m=>m, n=>n)
						port map(clk=>clk,
								en=>en,
								dataIn=>multiplyOut,
								dataOut=>yOut
	);
--
--	nodeProcess: process(clk)
--		
--	begin
--		if(clk'event and clk = '1') then
--			if(en = '1') then
--				
--			end if;
--		end if;
--	end process;

end architecture;