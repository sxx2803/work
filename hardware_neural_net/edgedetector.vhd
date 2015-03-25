library ieee;
use ieee.std_logic_1164.all;


entity edge_detector is
	generic (
		Nu : integer := 9;
		Nh : integer := 20;
		Ny : integer := 1;
		n  : integer := 7;
		m  : integer := 1
	);
	port (
		clk  : 	in std_logic;
		SE   : 	in std_logic;
		WSI  : 	in std_logic;
		b    : 	in std_logic_vector(m+n+1 downto 1);
		u    :  in std_logic_vector(Nu*(m+n+1) downto 1);
		edge : 	out std_logic
	);
end entity edge_detector;

architecture behavioral of edge_detector is
	
component node is
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
end component node;

component weightShift is  
	generic (
			nInputs   : integer := ((Nu+1)*Nh+((Nh+1)*Ny))*(m+n+1)
	);
  	port (
  		clk, WSI, SE : in  std_logic; 
        output      : out std_logic_vector(nInputs downto 1)
    ); 
end component weightShift; 

component comparator is  
	
	generic(nInputs         : integer);
  	port(
  		clk : in std_logic; 
  		input : in std_logic_vector(nInputs downto 1); 
        output    : out std_logic); 
end component comparator; 

component ionode is
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
end component;

signal sLayer1Output: std_logic_vector(Nu*(n+m+1) downto 1);
signal sLayer2Output: std_logic_vector(Nh*(n+m+1) downto 1);
signal sLayer3Output: std_logic_vector(Ny*(n+m+1) downto 1);
-- Since input weights are always 1.0, no need to scale it according to # of inputs
signal inputWeights: std_logic_vector(n+m+1 downto 1);
--signal hiddenWeights: std_logic_vector(Nu*Nh*(n+m+1) downto 1);
--signal outputWeights: std_logic_vector(Nh*Ny*(n+m+1) downto 1);
signal weights: std_logic_vector(((Nu+1)*Nh+((Nh+1)*Ny))*(m+n+1) downto 1);
signal sDoShit: std_logic;

signal sLayer1OutputWithBias: std_logic_vector((Nu+1)*(n+m+1) downto 1);
signal sLayer2OutputWithBias: std_logic_vector((Nh+1)*(n+m+1) downto 1);

constant numW : integer := ((Nu+1)*Nh+((Nh+1)*Ny));

signal bm    : std_logic_vector(m-1 downto 0) := (others => '0');
signal bn    : std_logic_vector(n-1 downto 0) := (others => '0');
	
begin
	
	bm <= (others => '0');
	bn <= (others => '0');
	inputWeights <= bm & "1" & bn;
	
	sDoShit <= not SE;
	sLayer1OutputWithBias <= b & sLayer1Output;
	sLayer2OutputWithBias <= b & sLayer2Output;
	
	weightShifter: weightShift 	generic map(nInputs=>((Nu+1)*Nh+((Nh+1)*Ny))*(m+n+1))
								port map(clk=>clk, WSI=>WSI, SE=>SE, output=>weights);
								
	comp         : comparator 	generic map(nInputs=>(Ny)*(m+n+1))
								port map(clk=>clk, input=>sLayer3Output, output=>edge);
	
	-- Generate input nodes
	Gen_Input_Nodes:
	for inIdx in Nu downto 1 generate
		inputNodeX: ionode 	generic map(numInputs => 1, m=>m, n=>n)
						 	port map(
						 		clk=>clk, 
						 		en=>sDoShit, 
						 		weights=>inputWeights, 
						 		xIn=>u(inIdx*(n+m+1) downto (inIdx-1)*(n+m+1)+1), 
						 		yOut=> sLayer1Output(inIdx*(n+m+1) downto (inIdx-1)*(n+m+1)+1)
						 );
	end generate Gen_Input_Nodes;
	
	-- Generate hidden layer nodes
	Gen_Hidden_Nodes:
	for hdnIdx in Nh downto 1 generate
		hiddenNodeX: node generic map(numInputs => (Nu+1), m=>m, n=>n)
						  port map(
						  		clk=>clk,
						  		en=>sDosHit,
						  		weights=>weights((((Nu+1)*hdnIdx+((Nh+1)*Ny)) * (n+m+1)) downto (((Nu+1)*(hdnIdx-1)+((Nh+1)*Ny)) * (n+m+1)) + 1),
						  		xIn => sLayer1OutputWithBias,
						 		yOut => sLayer2Output((hdnIdx)*(n+m+1) downto ((hdnIdx)-1)*(n+m+1)+1)
						  );
	end generate Gen_Hidden_Nodes;
	
	-- Generate output nodes
	Gen_Output_Nodes:
	for outIdx in Ny downto 1 generate
		hiddenNodeX: ionode generic map(numInputs => (Nh+1), m=>m, n=>n)
						  	port map(
						  		clk=>clk,
						  		en=>sDosHit,
						  		weights=>weights(((((Nh+1)*outIdx)) * (n+m+1)) downto ((((Nh+1)*(outIdx-1))) * (n+m+1)) + 1),
						  		xIn=> sLayer2OutputWithBias,
						 		yOut=> sLayer3Output(outIdx*(n+m+1) downto (outIdx-1)*(n+m+1)+1)
						  );
	end generate Gen_Output_Nodes;
	
--	mainProcess: process(clk)
--		
--	begin
--		if(clk'event and clk = '1') then
--			
--		end if;
--	end process mainProcess;

end architecture behavioral;
