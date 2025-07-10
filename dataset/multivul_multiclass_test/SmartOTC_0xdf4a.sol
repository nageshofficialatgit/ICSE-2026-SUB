// SPDX-License-Identifier: MIT
pragma solidity ^0.8.5;

/*
 * Copyright (c) 2024 CrypTop.dev
 */

interface Money20 {

    //function name() external view returns (string memory);

    function symbol() external  view returns (string memory);

    function decimals() external view returns (uint8);

    //function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

    //function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

interface AnySwap{
	function queryOrders(address money, uint160 chainCoin,uint start,uint queryType,address user,uint gtMoneyAmt,uint ltMoneyAmt,uint gtPrice18,uint ltPrice18,uint timeBefore, uint limit) view external returns( uint[] memory orders ) ;
	function addBuyOrder(address money,uint moneyAmt, uint160 chainCoin, uint amount8, uint8 times,string memory buyerCoinAddress) external payable returns(uint);
	function addSellOrder(address money, uint moneyAmt, uint160 chainCoin,uint amount8, uint8 times) external payable returns (uint);
	function lockBuyOrder(address money, uint160 chainCoin,uint orderkey) external payable returns(bool);
	function lockSellOrder(address money,uint160 chainCoin,uint orderkey,string memory buyerCoinAddress) external payable returns(bool);
	function buyerReceivedCoins(address money,uint160 chainCoin,uint orderkey) external returns (bool,bool);
	function cancelOrder(address money,uint160 chainCoin,uint orderkey) external returns (bool);
    function getOrder(address money, uint160 chainCoin, uint orderkey)
                view external returns (address seller,uint96 amount8,address buyer
                        , uint136 moneyAmt, uint8 times,string memory coinAddr, uint64 sellTime, uint64 buyTime );

	event AddBuyOrder(address sender,address money, uint160 chainCoin,uint orderkey);
	event AddSellOrder(address sender,address money,uint160 chainCoin,uint orderkey);
	event LockBuyOrder(address sender,address money,uint160 chainCoin,uint orderkey,uint moneyAmt,address buyer);
	event LockSellOrder(address sender,address money,uint160 chainCoin,uint orderkey,uint moneyAmt,address seller);
	event BuyerReceivedCoins(address sender,address money, uint160 chainCoin,uint orderkey,uint moneyAmt,address seller,bool,bool,string);
	event CancelOrder(address sender,address money,uint160 chainCoin,uint orderkey,bool,string);
	//event CancelSellOrder(address sender,address money,uint160 chainCoin,uint orderkey,bool,string);
}

contract SmartOTC is Money20, AnySwap{
	
	struct UserOrder {
		address seller;
		uint64 timeSell;
		
		uint96 amount8;
		uint136 moneyAmt;
		uint8 times;
        uint8 makerFlag;
		
		address buyer;
		uint64 timeBuy;
		
        string buyerCoinAddress;
    }

	address constant _maxquery = address(0x5f6d6178);//"_max";
	address constant _admin = address(0x5f61646d696e); //"_admin";
    uint160 constant _help = 0x5f68656c70;
    uint160 constant _sym = 0x5f73796d;

	uint8 public override constant decimals = 18;

	mapping( address =>
      	mapping( uint160 =>  
		      mapping( uint => UserOrder ) ) ) markets;

    mapping(address => uint) public minVolumes;
	mapping(address => uint) public serviceFees;
	mapping(uint160 => string) public texts;

	address public immutable owner ;

	modifier onlyAdmin() {
        require( msg.sender==owner || (minVolumes[_admin]!=0&&uint160(msg.sender) == minVolumes[_admin]), "Not admin");
        _;
    }

	//event Log(string,uint,address);

	
	constructor() payable {
		owner = msg.sender;
		//admin = msg.sender;

		//chains[_running] = 1; 
		minVolumes[_maxquery] = 30000; 
		texts[_sym] = "ETH";
        texts[_help] = "https://cryptop.dev/ for cross-chain swap. v1.0.4";

	}
    
    function help() external view returns (string memory txt)
                        {
        txt = texts[_help];

    }
    
    function symbol() external view override returns (string memory sym)
                        {
        sym = texts[_sym];

    }
 
	function transfer(address, uint256) pure external override returns (bool){
		return false;
	}
	
	function transferFrom(address , address , uint256 ) pure external override returns (bool){
		return false;
	}

	function allowance(address , address ) external pure override returns (uint256){
		return type(uint).max;
	}
    
    function balanceOf(address account) external view override returns (uint256){
        return account.balance;
    }

	function queryOrders(address money, uint160 chainCoin,uint start
		,uint queryType,address user,uint gtMoneyAmt,uint ltMoneyAmt
				,uint gtPrice18, uint ltPrice18, uint timeBefore, uint limit)
					view external override returns( uint[] memory keyvalues ) {
		
        mapping( uint => UserOrder ) storage market = markets[money][chainCoin];

		if(start == 0)
			start = market[0].amount8;

		if(limit>200)
			limit = 200;

		UserOrder memory uo;
		uint end = (start>minVolumes[_maxquery])?(start-minVolumes[_maxquery]):0;
		//uint size = limit*10;
		uint[] memory keys = new uint[](limit*10);
		uint ind;

		for( uint orderkey = start;  orderkey>end && ind<(limit*10); orderkey-=1 ) {
				uo = market[ orderkey ];
				if(uo.amount8==0){
					continue;
				}
				if(queryType==1){ //sell orders unlocked
					if(  uo.buyer!=address(0) ){
						continue;
					}
				}else if(queryType==2){ //buy orders unlocked
					if(  uo.seller!=address(0) ){
						continue;
					}
				}else if(queryType==12){ //unlocked
					if(uo.seller!=address(0)&&uo.buyer!=address(0))
						continue;
				}else if(queryType==10){ //locked
					if(uo.seller==address(0)||uo.buyer==address(0))
						continue;
				}
				if( user!=address(0) && uo.buyer!=user && uo.seller!=user ){
						continue;
				}
				if(gtMoneyAmt>0 && (uo.moneyAmt*uo.times)<gtMoneyAmt){
					continue;
				}
				if(ltMoneyAmt>0 && (uo.moneyAmt*uo.times)>ltMoneyAmt){
					continue;
				}
				if(ltPrice18>0 && (uint(uo.moneyAmt)*100000000/uo.amount8)>ltPrice18){
					continue;
				}
				if(gtPrice18>0 && (uint(uo.moneyAmt)*100000000/uo.amount8)<gtPrice18){
					continue;
				}
				if(timeBefore>0){
					if(uo.timeSell>timeBefore||uo.timeBuy>timeBefore)
							continue;
				}

				keys[ind] = orderkey;
				keys[ind+1] = uint160(uo.seller);
                keys[ind+1] = (keys[ind+1]<<64) + uo.timeSell;
				keys[ind+2] = uo.amount8;
                keys[ind+3] = uint160(uo.buyer);
                keys[ind+3] = (keys[ind+3]<<64) + uo.timeBuy;
				keys[ind+4] = uo.moneyAmt;
				keys[ind+5] = uo.times;
				bytes memory byts = bytes(uo.buyerCoinAddress);
				keys[ind+6] = uint(bytes32(byts));
				if(byts.length>32){
					//get byts[32:64]
					keys[ind+7] = uint(bytes32( bytesSlice(byts,32,64) ));
				}
				if(byts.length>64){
					//get byts[64:96]
					keys[ind+8] = uint(bytes32( bytesSlice(byts,64,96) ));
				}
                if(byts.length>96){
                    //get byts[96:128]
                    keys[ind+9] = uint(bytes32( bytesSlice(byts,96,128) ));
                }
				ind += 10;
		}

		keyvalues = new uint[](ind);
		for(uint32 i=0; i<ind;i++){
			keyvalues[i] = keys[i];
		}
		
	 }

	function bytesSlice(bytes memory byts,uint start, uint end) internal pure returns (bytes memory newbyts){
		for(uint i=start;i<byts.length && i<end;i++){
			newbyts = bytes.concat(newbyts,byts[i]);
		}
		
	}

	function orderDeposit(address money,address from, uint amount) internal{
		//emit Log("Deposit amount",amount,address(0));
		if(money == address(this)){
			require(msg.value>=amount, "Deposit failed0");
			if(msg.value>amount){
				payable(from).transfer(msg.value-amount);
			}
		}else{
			require(msg.value==0, "Deposit failed1");
			//Money20 mny = Money20(money);
			//bool succ = mny.transferFrom(from,address(this),amount);
            (bool success, bytes memory data) = money.call(abi.encodeWithSelector(0x23b872dd, from, address(this),amount));
            require(success && (data.length == 0 || abi.decode(data, (bool))), "Deposit failed2");
		}

	}
	 
	function addBuyOrder( address money, uint moneyAmt
						, uint160 chainCoin, uint amount8, uint8 times,string memory buyerCoinAddress)
				payable public override returns( uint orderkey ) {
        require( bytes(buyerCoinAddress).length<129, "Address too long" );
		uint minvol = minVolumes[money];
		require( minvol > 1 ,"Money invalid" );
		require( times > 0 ,"Zero times" );

        require( amount8 > 0,"Zero amount");
		require( moneyAmt >= minvol,"Tiny order");
		
		orderDeposit(money,msg.sender,moneyAmt+moneyAmt);

		//buy market
        mapping( uint => UserOrder ) storage market = markets[money][chainCoin];

		orderkey = market[0].amount8;
		orderkey += 1;
		market[0].amount8 = uint96(orderkey);

		UserOrder storage ord = market[orderkey];
		ord.amount8 = uint96(amount8);
		ord.moneyAmt = uint136(moneyAmt);

		ord.times = times;
		ord.buyer = msg.sender;

		ord.timeBuy = uint64(block.timestamp);
		if(bytes(buyerCoinAddress).length>0)
		   ord.buyerCoinAddress = buyerCoinAddress;

		emit AddBuyOrder(msg.sender,money, chainCoin, orderkey);

	}
	

	function lockBuyOrder(address money, uint160 chainCoin, uint orderkey)
							payable public override returns(bool succ) {

			UserOrder storage uo = markets[money][chainCoin][orderkey];

			if(uo.seller == address(0) && uo.buyer!=address(0) ){
				uint mamount = uo.moneyAmt;

				require(mamount>0,"Missed order"); //finished
				orderDeposit(money,msg.sender,mamount);
				uo.seller = msg.sender;
				uo.timeSell = uint64(block.timestamp);
                succ = true;
				
				emit LockBuyOrder(msg.sender,money, chainCoin, orderkey,mamount,uo.buyer);
			}else{
				//return msg.value
				revert("Others locked");
			}			

	}


	function buyerReceivedCoins(address money,uint160 chainCoin,uint orderkey) 
                            public override returns (bool succSeller, bool succBuyer){
			mapping( uint => UserOrder ) storage market = markets[money][chainCoin];
			UserOrder memory uo = market[orderkey];

			require(msg.sender!=address(0)&&uo.buyer == msg.sender,"Not buyer");
			require(uo.seller != address(0) ,"Unlocked");

            //uint fee = uo.moneyAmt/100;
			uint buyeramt = uo.moneyAmt;
			uint selleramt = uo.moneyAmt+uo.moneyAmt;
			uint servfee = uo.moneyAmt/100;
                                   
            if(uo.makerFlag == 0){
                selleramt -= servfee;
            }else{
                buyeramt -= servfee;
            }
			
			delete market[orderkey]; //reEntry guard

			succSeller = moneySend(money,uo.seller, selleramt);
			succBuyer = moneySend(money,uo.buyer,buyeramt);

			if(!succSeller)
				servfee += selleramt;

			if(!succBuyer)
				servfee += buyeramt;
			
			//if(servfee>0)
				serviceFees[money] += servfee;

			emit BuyerReceivedCoins(msg.sender,money, chainCoin, orderkey,uo.moneyAmt,uo.seller
			,succSeller,succBuyer,(succSeller&&succBuyer)?"":"Failed funds to admin");
			
	}

	function moneySend(address token,address rcv,uint amt) 
			internal returns(bool succ){
			if(token == address(this))
				return payable(rcv).send(amt);
			else{
                (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0xa9059cbb, rcv,amt));
                succ = success && ( data.length == 0 || abi.decode(data, (bool)) ) ;
            /*
				Money20 mny = Money20(token);
				try mny.transfer(rcv,amt) returns (bool suc){
					succ = suc;
				}catch{
					succ = false;
				}
             */
			}
				
	}


	function addSellOrder(address money, uint moneyAmt, uint160 chainCoin, uint amount8, uint8 times)
				payable public override returns( uint orderkey ) { 
		uint minvol = minVolumes[money];
		require( minvol > 1 ,"Money invalid" );
		require(times>0,"Zero times");

        require( amount8 > 0,"Zero amount");
		require( moneyAmt >= minvol,"Tiny order");

		orderDeposit(money,msg.sender, moneyAmt);

		//sell market
        mapping( uint => UserOrder ) storage market = markets[money][chainCoin];

		orderkey = market[0].amount8;
		orderkey += 1;
		market[0].amount8 = uint96(orderkey);
		
		UserOrder storage ord = market[orderkey];
		ord.amount8 = uint96(amount8);
		ord.moneyAmt = uint136(moneyAmt);
		ord.times = times;
        ord.makerFlag = 1;
		ord.seller = msg.sender;
		ord.timeSell = uint64(block.timestamp);
		//market[orderkey] = ord;

		emit AddSellOrder(msg.sender,money, chainCoin, orderkey);

	}
	

	function lockSellOrder(address money, uint160 chainCoin, uint orderkey
										 , string memory buyerCoinAddress)
						payable public override returns(bool succ) {
            require( bytes(buyerCoinAddress).length<129, "Address too long" );
			UserOrder storage uo = markets[money][chainCoin][orderkey];

			if(uo.buyer == address(0) && uo.seller!=address(0) ){
				uint mamount = uo.moneyAmt;
				require(mamount>0,"Missed order"); //finished
				orderDeposit(money, msg.sender,mamount+mamount);
				uo.buyer = msg.sender;
				uo.timeBuy = uint64(block.timestamp);
				if(bytes(buyerCoinAddress).length>0)
					uo.buyerCoinAddress = buyerCoinAddress;
                succ = true;
				
				emit LockSellOrder(msg.sender,money, chainCoin, orderkey,mamount,uo.seller);
			}else{
				revert("Others locked");
			}		

	}


	function cancelOrder(address money, uint160 chainCoin, uint orderkey)
				public override returns (bool succ) {
            require(msg.sender!=address(0), "Unknown user");
			mapping( uint => UserOrder ) storage market = markets[money][chainCoin];
			UserOrder memory uo = market[orderkey];
            uint refundamt = uo.moneyAmt;
            require( refundamt>0 ,"Zero amount");
            
            if(uo.buyer == address(0)){ //sell order
                    require( uo.seller==msg.sender ,"Not seller");
                                   
            }else if(uo.seller == address(0)){ //buy order
                    require( uo.buyer==msg.sender, "Not buyer");
                    refundamt += refundamt;
            }else{
                    revert("Order locked");
            }
                                   
			delete market[orderkey];//reEntry guard
			succ = moneySend(money,msg.sender, refundamt);
            string memory txt;

			if(!succ){
				serviceFees[money] += refundamt;
				txt = "Failed funds to admin";
			}

			emit CancelOrder(msg.sender,money, chainCoin, orderkey,succ,txt );

	}
                                   

                                   
    function getOrder(address money, uint160 chainCoin, uint orderkey)
                    view public override returns (address seller,uint96 amount8,address buyer
                            , uint136 moneyAmt, uint8 times,string memory coinAddr, uint64 sellTime, uint64 buyTime ) {

        UserOrder memory uo = markets[money][chainCoin][orderkey];
        seller = uo.seller;
        amount8 = uo.amount8;
        buyer = uo.buyer;
        moneyAmt = uo.moneyAmt;
        times = uo.times;
        coinAddr = uo.buyerCoinAddress;
        sellTime = uo.timeSell;
        buyTime = uo.timeBuy;

    }
                                   
	
	function string2hex(string memory str) pure public returns(bytes memory){
		
		return bytes(str);
		
	}
                                   
    function hex2string(bytes memory byts) pure public returns(string memory){

        return string(byts);
                                       
    }

	function setMinVolume(address money, uint vol) 
										onlyAdmin public {
        minVolumes[money] = vol;
	}
                                   

	function setText(uint160 key,string memory txt)
	             					  	onlyAdmin public{
		texts[key] = txt;

	}

	function withdrawServiceFee(address money)
		        	            	onlyAdmin public {
			uint fee = serviceFees[money];
			uint remain = fee/10;
			require(remain>0,"Tiny order");
            serviceFees[money] = remain; //keep not zero
			bool succ = moneySend(money,msg.sender,fee-remain);
			require(succ,"Failed funds to admin");

	}


}