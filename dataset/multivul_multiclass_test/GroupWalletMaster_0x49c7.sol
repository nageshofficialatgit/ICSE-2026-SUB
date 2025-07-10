// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.18 <0.8.20;

// ungravel.eth, GroupWalletFactory, GroupWalletMaster, GroupWallet, ProxyWallet, TokenMaster, ProxyToken, PrePaidContract, AuctionMaster, BiddingProxy, intDeedMaster, extDeedMaster, IntDeedProxy, Intentions by pepihasenfuss.eth 2017-2025, Copyright (c) 2025

// GroupWalletMaster is based on MultiSigContracts inspired by Parity MultiSignature contract, consensys and gnosis MultiSig contracts.
// GroupWallet and Ungravel is entirely based on Ethereum Name Service, "ENS", the domain name registry.

//   ENS, ENSRegistryWithFallback, PublicResolver, Resolver, FIFS-Registrar, Registrar, AuctionRegistrar, BaseRegistrar, ReverseRegistrar, DefaultReverseResolver, ETHRegistrarController,
//   PriceOracle, SimplePriceOracle, StablePriceOracle, ENSMigrationSubdomainRegistrar, CustomRegistrar, Root, RegistrarMigration are contracts of "ENS", by Nick Johnson and team.
//
//   Copyright (c) 2018, True Names Limited / ENS Labs Limited
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

abstract contract AbstractENS {                                                                    // ENS Registry grants access to domain names and domain name properties
  event NewOwner(bytes32 indexed node, bytes32 indexed label, address ownerENS);
  event Transfer(bytes32 indexed node, address ownerENS);
  event NewResolver(bytes32 indexed node, address resolver);
  event NewTTL(bytes32 indexed node, uint64 ttl);
  event ApprovalForAll(address indexed ownerENS, address indexed operator, bool approved);

  function setSubnodeRecord(bytes32 nodeENS, bytes32 label, address ownerENS, address resolver, uint64 ttl) external virtual;
  function setOwner(bytes32 nodeENS, address ownerENS) external virtual;
  function owner(bytes32 nodeENS) external view virtual returns (address);
  function recordExists(bytes32 nodeENS) external view virtual returns (bool);
}

abstract contract AbstractGroupWalletFactory {                                                     // Group Wallet Factory, GWF, main Ungravel entry point coordinating Ungravel Groups and all activities, deploying ProxyGroupWallet, GWP, and ProxyToken, aka TokenProxy
  AbstractResolver                public  resolverContract;
  AbstractETHRegistrarController  public  controllerContract;
  AbstractGWMBaseRegistrar        public  base;
  AbstractENS                     public  ens;
  AbstractReverseRegistrar        public  reverseContract;
  Abstract_NameWrap               public  ensNameWrapper;

  function getProxyToken(bytes32 _domainHash) external view virtual returns (address p);
  function reserve_replicate(bytes32 _domainHash,bytes32 _commitment) external virtual payable;
  function replicate_group_l9Y(bytes32[] calldata _m, bytes calldata data32, bytes32[] calldata _mem) external virtual payable;
  function upgradeGWF(bytes32 _dHash, address _GWF) external virtual payable;
  function tld() public view virtual returns (string memory);
  function receivePayment() external virtual payable returns(bool);
  function getOwner(bytes32 _domainHash) external virtual view returns (address);
  function _register(bytes32 _dHash,uint256 _rent,string memory _name,uint256 _dur,bytes32 _secret) external virtual payable;
}

abstract contract AbstractTokenProxy {                                                             // TokenProxy gives access to the Group Shares contract, aka TokenProxy contract, that comes with each group, a proxy contract that belongs to TokenMaster
  function balanceOf(address tokenOwner) external virtual view returns (uint thebalance);
  function drainShares(bytes32 dHash, address from, address toReceiver) external virtual;
  function drainLegacyShares(bytes32 dHash, address from, address toReceiver) external virtual;
  function approve_v2d(address spender, uint tokens) external virtual;
  function transferFrom_78S(address from, address toReceiver, uint amount) external virtual;
  function tokenAllow(address tokenOwner,address spender) external virtual view returns (uint256 tokens);
  function transfer_G8l(address toReceiver, uint amount) external virtual;
  function sell_LA2(uint256 amount) external virtual;
  function upgradeTokenMaster(bytes32 dHash, address master) external payable virtual;
  function upgradeGWM(bytes32 dHash, address master) external payable virtual;
  function substring(bytes memory self, uint offset, uint len) public pure virtual returns(bytes memory);
}

contract AbstractGWMBaseRegistrar {                                                                // BaseRegistrar belongs to the ENS - Ethereum Naming Service
  event NameMigrated(uint256 indexed id, address indexed ownerB, uint expires);
  event NameRegistered(uint256 indexed id, address indexed ownerB, uint expires);
  event NameRenewed(uint256 indexed id, uint expires);

  bytes32 public baseNode;   // The namehash of the TLD this registrar owns (eg, .eth)
}

abstract contract AbstractResolver {                                                               // ENS Resolver provides the address and properties of domain names, s.a. "your-company.base", it resolves domain names to EVM addresses
  event AddrChanged(bytes32 indexed node, address a);
  event AddressChanged(bytes32 indexed node, uint coinType, bytes newAddress);
  event NameChanged(bytes32 indexed node, string name);
  event ABIChanged(bytes32 indexed node, uint256 indexed contentType);
  event PubkeyChanged(bytes32 indexed node, bytes32 x, bytes32 y);
  event TextChanged(bytes32 indexed node, string indexed indexedKey, string key);
  event ContenthashChanged(bytes32 indexed node, bytes hash);

  function setABI(bytes32 nodeC, uint256 contentType, bytes calldata data) external virtual;
  function setAddr(bytes32 nodeD, address r_addr) external virtual;
  function setName(bytes32 nodeF, string calldata _name) external virtual;
  function setText(bytes32 nodeG, string calldata key, string calldata value) external virtual;
  function supportsInterface(bytes4 interfaceId) external virtual view returns (bool);
}

abstract contract Abstract_NameWrap {                                                              // NameWrapper is used by the ENS on Ethereum mainnet or Sepolia, only needed on ENS3 compatible chains
  function setSubnodeRecord(bytes32 parentNode,string memory label,address owner,address resolver,uint64 ttl,uint32 fuses,uint64 expiry) external virtual returns (bytes32 node);
  function setSubnodeOwner(bytes32 node,string calldata label,address newOwner,uint32 fuses,uint64 expiry) external virtual returns (bytes32);
  function ownerOf(uint256 id) external virtual view returns (address);
  function setApprovalForAll(address operator,bool approved) external virtual;
}

interface I_NameWrapper {                                                                          // NameWrapper is used by the ENS on Ethereum mainnet or Sepolia, only needed on ENS3 compatible chains
  function setSubnodeRecord(bytes32 parentNode,string memory label,address owner,address resolver,uint64 ttl,uint32 fuses,uint64 expiry) external returns (bytes32 node);
  function setSubnodeOwner(bytes32 node,string calldata label,address newOwner,uint32 fuses,uint64 expiry) external returns (bytes32);
  function ownerOf(uint256 id) external view returns (address);
  function setApprovalForAll(address operator,bool approved) external;
}

abstract contract AbstractETHRegistrarController {                                                 // RegistrarController belongs to ENS, it handles the purchase and rent of domain names, s.a. "my-company.scroll" | "your-company.blast"
  event NameRegistered(string name, bytes32 indexed label, address indexed ownerRC, uint cost, uint expires);
  event NameRenewed(string name, bytes32 indexed label, uint cost, uint expires);
  address public nameWrapper;
  function rentPrice(string memory name, uint duration) view external virtual returns(uint);
  function available(string memory name) external virtual view returns(bool);
  function registerWithConfig(string memory name, address owner, uint duration, bytes32 secret, address resolver, address addr) external virtual payable;
}

abstract contract AbsETHRegV3 {                                                                    // ETHRegistrar is only needed with isENSV3() on ENS - chains, such as Ethereum mainnet or Sepolia, integrating with the original ENS installation 
  address public nameWrapper;
  function rentPrice(string memory name, uint duration) view external virtual returns(uint,uint);
}

abstract contract AbstractGroupWalletProxy {                                                       // Abstract and access to GWP, Group Wallet Proxy contract, the Voting and Multi-Sig-contract of each group, a proxy, belonging to the GroupWallet Master
  function submitFirstTransaction_gm(uint firstTRecord, uint256 dhash) external virtual payable;
  function submitLegacyTransaction(uint tNb,uint tRecord) external virtual payable;
  function getMasterCopy() external view virtual returns (address);
}

abstract contract Resolver {                                                                       // ENS Resolver provides the address and properties of domain names, s.a. "your-company.base"
  mapping (bytes32 => string) public name;
}

abstract contract AbstractReverseRegistrar {                                                       // Reverse Resolver and Reverse Default Resolver give access to the domain name, if only an address is given
  Resolver public defaultResolver;
  function node(address addr) external virtual pure returns (bytes32);
}

abstract contract AbstractAuctionRegistrar {                                                       // Auction Registrar is Vickrey Auction, controlled by Ungravel GroupWallets, in order to sell group shares to investors and to tame competition
  function startAuction_ge0(bytes32 _hash, uint revealP) public virtual payable;
}

abstract contract AbsInt {                                                                         // Intentions store and provide information about LOIs. LOIs are also important in order to participate in and to profit from Ungravel Global Token, aka Ungravel Global Shares, in the future.
  function saveLetterOfIntent(address target, uint nbOfShares) public virtual payable;
  function storeInvestment(uint _nbOfShares, address _gwp, uint _pricePaidEth, bytes32 _salt) public virtual payable;
}

abstract contract AbsGlobalT {
  function convertGroupSharesToEther(address _groupTokenContract, uint gShares) public virtual payable;
}

abstract contract AbsGWM {                                                                         // GroupWallet Master is the Master contract of any GroupWalletProxy contract, GWP
  function getIntention() public virtual view returns (AbsInt);
  function proxyToken() public virtual payable returns (address);
}

abstract contract ABSReg {                                                                         // Auction Registrar, aka AuctionMaster
  function getGWPfromAuction(bytes32 _hash) public virtual view returns (address);
}

abstract contract ABSExtDeed {                                                                     // external Deeds are BidBuckets proxy contracts, with a extDeedMaster as a Master
  address public  owner;
  ABSReg  public  registrar;
  bytes32 public  lhash;
}


contract GroupWalletMaster {                                                    // GroupWalletMaster, GWM, is the Master of all GWP,  GroupWalletProxy contracts, mainly for voting and the multi-sig functionality
    address internal masterCopy;                                                // ProxyGroupWallet needs this ******* DO NOT CHANGE ORDER ******

    mapping(uint256 => uint256) private  tArr;                                  // transaction records, 32 bytes to store any transaction (saves transaction fees)
    address[]                   private  owners;                                // contract owners = members of the group (2-31)

    address internal GWF;                                                       // GWF - GroupWalletFactory contract, main entry point for ungravel
    mapping(uint256 => bytes)   private  structures;                            // saving encoded group structure for addUser() or submitted proposals, temporarily used
    
    address internal intent;                                                    // Intention Contract 
                                                                                // ******* DO NOT CHANGE ORDER ******

    // Master contract for group wallets, a low-cost x-chain group wallet with multi-sig, group shares and Funding Auctions

    event TestReturn(uint256 v1, uint256 v2, uint256 v3, uint256 v4);
    event GroupWalletDeployed(address sender, uint256 members, uint256 timeStamp);
    event Deposit(address from, uint256 value);
    event ColorTableSaved(bytes32 domainHash);
    event EtherScriptSaved(bytes32 domainHash,string key);

    // ----------------------  GWM ---------------------------------------------

    function getMasterCopy() public view returns (address) {
      return masterCopy;
    }

    function getGWF() external view returns (address) {
      return GWF;
    }
    
    function getENS() internal view returns (AbstractENS) {
      return AbstractENS( AbstractGroupWalletFactory(GWF).ens() );
    }

    function getRsv() internal view returns (AbstractResolver) {
      return AbstractGroupWalletFactory(GWF).resolverContract();
    }

    function getCtrl() internal view returns (AbstractETHRegistrarController) {
      return AbstractGroupWalletFactory(GWF).controllerContract();
    }
    
    function getRevs() internal view returns (AbstractReverseRegistrar) {
      return AbstractGroupWalletFactory(GWF).reverseContract();
    }

    function getNWrap() internal view returns (I_NameWrapper) {
      return I_NameWrapper(getCtrl().nameWrapper());
    }

    function getIntention() public view returns (AbsInt) {
      if (intent!=address(0x0)) return AbsInt( intent );
      return AbsInt(AbsGWM(getMasterCopy()).getIntention());
    }

    function getTarget(uint tNb) private view returns (address) {
      return address( uint160( uint256( uint256( tArr[tNb] ) & k_addressMask ) ) );
    }

    function getTValue(uint tNb) private view returns (uint64) {
      return uint64( uint256( (uint256( uint256( tArr[tNb] ) & k_valueMask )>>160) & k_value2Mask ) );
    }

    function getAsset(uint tNb) private view returns (uint8) {
      return uint8( uint256( uint256(uint256( uint256( tArr[tNb] ) & k_assetMask )>>208) & k_asset2Mask ) );
    }
    
    function getAssetT(uint256 t) private pure returns (uint8) {
      return uint8 (uint256( (uint256( uint256( t ) & k_assetMask )>>208) & k_asset2Mask ));
    }
    
    function getFlags(uint tNb) private view returns (uint64) {
      return uint64( uint256( uint256(uint256( uint256( tArr[tNb] ) & k_flagsMask )>>216) & k_flags2Mask ) );
    }
    
    function getFlagsT(uint256 t) private pure returns (uint64) {
      return uint64( (uint256( uint256( t ) & k_flagsMask )>>216) & k_flags2Mask );
    }

    function getType(uint tNb) private view returns (uint8) {
      return uint8( uint256( uint256(uint256( uint256( tArr[tNb] ) & k_typeMask )>>252) & k_type2Mask ) );
    }

    function getStructure(uint256 target) external view returns (bytes memory) {
      return structures[target];
    }

    function saveFlags(uint _tId, uint64 flags) private {
      tArr[_tId] = uint256( uint256(uint256( flags )<<216) & k_flagsMask ) + uint256( tArr[_tId] & k_flags3Mask );
    }
    
    function saveAsset(uint _tId, uint8 asset) private {
      tArr[_tId] = uint256( uint256(uint256( asset )<<208) & k_assetMask ) + uint256( tArr[_tId] & k_asset3Mask );
    }
    
    function saveExecuted(uint _tId, uint64 f, uint t) private {
      tArr[_tId] = uint256( uint256( uint256((uint256(f)<<216) & k_flagsMask) + uint256( t & k_flags3Mask ) ) | k_executeFlag );
    }
    
    function saveNotExecuted(uint _tId, uint64 f, uint t) private {
      tArr[_tId] = uint256( ((uint256(f)<<216) & k_flagsMask) + uint256( t & k_flags3Mask ) );
    }
    
    function getBase() internal view returns (bytes32) {
      return AbstractGroupWalletFactory(GWF).base().baseNode();
    }
    
    function getPtoken(bytes32 _hash) internal view returns (AbstractTokenProxy) {
      return AbstractTokenProxy( AbstractGroupWalletFactory(GWF).getProxyToken( _hash ) );
    }
    
    function getHash() internal view returns (bytes32) {
      return bytes32( tArr[uint256(uint160(GWF))] );  // *** ATTENTION: This requires at least one transaction ***
    }
    
    function dHashFromLabelBytes32(bytes32 _name) internal view returns (bytes32 hsh) {
      return keccak256(abi.encodePacked( getBase(), keccak256(bytes(bytesToStr(_name, uint(_name)&0xff))) )); // dhash e.g. group-rebels.eth
    }
    
    function adjustMajority(uint64 _f) internal view returns (uint64) {
      return (uint64(_f|uint64(MAX_OWNER_COUNT)) ^ uint64(MAX_OWNER_COUNT))+(uint64(activeOwners())/2)+1;
    }

    // -------------------  strings --------------------------------------------

    function my_require(bool b, string memory str) private pure {
      require(b,str);
    }
    
    function bytesToStr(bytes32 _b, uint len) internal pure returns (string memory) {
      bytes memory bArr = new bytes(len);
      uint256 i;
      
      do
       { 
        bArr[i] = _b[i];
        i++;
      } while(i<len&&i<32);
      
      return string(bArr); 
    }
    
    function bytes32ToStr(bytes32 _b) internal pure returns (string memory) {
      bytes memory bArr = new bytes(32);
      uint256 i;
      
      uint off = 0;
      do { 
        if (_b[i] != 0) bArr[i] = _b[i];
        else off = i;
        i++;
      } while(i<32&&off==0);
      
      
      bytes memory rArr = new bytes(off);
      
      i=0;
      do
       { 
        if (bArr[i] != 0) rArr[i] = bArr[i];
        off--;
        i++;
      } while(i<32&&off>0);
      
      return string(rArr); 
    }
    
    function toLowerCaseBytes32(bytes32 _in) internal pure returns (bytes32 _out){
      if ( uint256(uint256(uint256(_in) & k_typeMask) >> 252) < 6 ) return bytes32(uint256(uint256(_in) | 0x2000000000000000000000000000000000000000000000000000000000000000 ));
      return _in;
    }
    
    function mb32(bytes memory _data) private pure returns(bytes32 a) {
      // solium-disable-next-line security/no-inline-assembly
      assembly {
          a := mload(add(_data, 32))
      }
    }
    
    function labelStrFromBytes32(bytes32 _label) private pure returns (string memory) {
      return bytesToStr(_label,uint(_label)&0xff);
    }
    
    // -------------------  transactions ---------------------------------------
    
    function checkPreconditions(uint tNb, uint tRecord, uint cmd) private view {
      my_require(tNb>0&&isAddressOwner(msg.sender)&&tArr[tNb]==0&&uint(uint256(tRecord & k_typeMask)>>252)==cmd,"V");
    }

    function isLegacyTokenCmd(uint tRecord) private pure returns (bool) {
      return uint8( uint256( uint256(uint256( uint256( tRecord ) & k_typeMask )>>252) & k_type2Mask ) )==k_legacyToken;
    }
    
    function checkSplitPreconditions(uint tNb, uint8 exec) private view returns (uint) {
      my_require(owners.length>3,"G");
      return getOpenSplitTransactionNb(tNb-1,exec);
    }
    
    function reverseDNameFromAddress(address a) internal view returns (string memory) {
      return getRevs().defaultResolver().name( getRevs().node(a) );
    }
    
    // -------------------  multiSig wallet ------------------------------------

    function confirmTransaction_Q6d(uint _tId) external payable {
      if (_tId>0) my_require(msg.value>0,"Q");    
      uint256 t = tArr[_tId];
      uint64 f  = getFlagsT(t);
      uint64 o  = getOwnerMask(msg.sender);
      my_require(uint64(f&o)==0&&getAssetT(t)<128,"X");
      
      f = uint64(f|o);                                                          // confirm f|o

      if ( ( getFlags(msg.value-1) & uint64(MAX_OWNER_COUNT) ) <= nbOfConfirmations(f) ) executeMultiSigCommand(_tId,t,f);
      else saveNotExecuted(_tId, f, t);
    }

    function revokeConfirmation_NlP(uint _tId) external {
      uint256 t = tArr[_tId];
      uint64 f  = uint64(uint256( uint256(uint256( uint256( t ) & k_flagsMask )>>216) & k_flags2Mask));
      uint64 o  = getOwnerMask(msg.sender);

      my_require(o>0&&uint64(f&o)>0&&(uint256(t)&k_executeFlag)==0,"r");
  
      tArr[_tId] = uint256( (uint256( uint64(f|o) ^ uint64(o) )<<216) & k_flagsMask ) + uint256( t & k_flags3Mask );
    }
  
    function storeLegacyProxyToken(bytes32 spinName,uint64 flags) internal returns (address) {
      bytes32 newHash = keccak256( abi.encodePacked( getBase(), keccak256( bytes( bytes32ToStr( toLowerCaseBytes32(spinName) ) ) ) ) );     // dhash e.g. group-rebels.eth
      address gwp = getENSOwnerExpected( newHash );                                                                                         // GroupWalletProxy contract address of spin-off group
      
      address legacyToken = proxyToken();                                                                                                   // legacyToken contract of legacy group before spin-off
      uint256 tRecord     = uint256(uint256(uint256(k_legacyToken & k_type2Mask)<<252)&k_typeMask) + uint256(uint256(uint256(flags)<<216)&k_flagsMask) + uint256( uint256(uint256( nbOfConfirmations(flags) )<<208) & k_assetMask ) + uint256(uint160(legacyToken)) | k_executeFlag; // t type == 0xb, store legacy tokenContract for legacy assets

      AbstractGroupWalletProxy(gwp).submitFirstTransaction_gm{value:0}(tRecord, uint256(newHash));
      
      return gwp;
    }
    
    function sendSpinoffAssets(uint _tId, uint64 f, bytes32[] calldata _mem, uint totalGas) internal returns (bool) {
      uint cost = uint256(msg.value);
      uint64 remainingRequired;

      // ------- shares to new addresses, drain legacy shares ------------------

      {
        uint64  o  = 1;                                                         // drainShares to leaving members
        uint64 f_s = 0;
        uint256 t  = 0;
        uint    i  = 0;
        bytes32 hash = getHash();
        do {
          if ((uint64(f>>5)&o)==o) {
            t =  uint256(_mem[7+(f_s*5)]) & k_addressMask;                      // receiver
            f_s++;
            
            getPtoken(hash).drainShares(hash,owners[i],address(uint160(t)));
            
            drainLegacySharesGWP(hash,i,address(uint160(t)));
            
            ownerChange(owners[i], address(0x0));                               // remove spin-off group member
          }
          
          o = o*2;
          i++;
        } while(i<owners.length);
        
        remainingRequired = uint64((activeOwners()/2)+1);

        f_s = uint64(uint64(f|uint64(MAX_OWNER_COUNT)) ^ uint64(MAX_OWNER_COUNT))+remainingRequired; // reset majority spinOff CMD
        saveFlags(_tId,f_s);
      }

      // --------------------- compute member share of cost  -------------------
      
      uint nbm  = uint64((_mem.length-5)/5);                                    // nb of members of spinoff group

      {
        cost    += uint(uint256((totalGas-gasleft())-uint(23504*nbm))*tx.gasprice);// cost of this spinoff transaction in wei
        cost     = cost/nbm;                                                    // cost per member

        //emit TestReturn(uint256 ( msg.value ), uint256 (payable(this).balance), uint256 ( remainingRequired ), uint256( cost ) );
      }
      
      
      uint64 ownerId = 0;
    
      // ---------------- store one legacy token contract adddress -------------
      
      address gwp = storeLegacyProxyToken(_mem[3],f);                           // spin-off group name, spinOff flags

      {                                                                         // pay back deposits to leaving members (ex-members)
        uint flag = uint64(uint64(2**nbm)-1);                                   // 2**nb -1 = all members mask, e.g. 0x7 for 3 members
        uint i    = 0;                                                          // index i                          
        uint64 o;                                                               // owner ID
        uint256 t = 0;                                                          // transaction record
        address depositReceiver;                                                // member of spinOff group that receives its deposit back
        uint    depositValue;
        
        do {                                                                    // iterate transaction records t
          t = tArr[i];
          
          if ((t&k_executeFlag==0) && ((t & k_typeMask)>>252==k_deposit)) {     // cmd type = 10 = deposit ether/funds, not yet executed
            o = uint8((uint256( uint256(t)&k_assetMask )>>208)&k_asset2Mask);   // nb/id of new owner
            
            depositReceiver = address(uint160(uint256(_mem[7+(o*5)]) & k_addressMask));
            depositValue    = uint(uint256(t) & k_addressMask);
            
            if ( (cost>0) && (depositValue > cost) && (payable(this).balance >= (depositValue-cost)) ) {
              my_require(payable(depositReceiver).send(depositValue-cost),"L");  
              emit Deposit(depositReceiver,depositValue-cost);
            
              tArr[i] = uint256(uint256( uint256(uint256( remainingRequired )<<216) & k_flagsMask ) + uint256( t & k_flags3Mask )) | k_executeFlag; // mark deposit DONE and paid back
            }
            
            flag = flag ^ uint64(2**uint64(o));                                 // computes new member nb of sender in spinoff group
          }
          
          // ---------------- store older legacy token contracts ---------------
          
          if (((t & k_typeMask)>>252)==k_legacyToken) AbstractGroupWalletProxy(gwp).submitLegacyTransaction{value:0}(i+1,t);  // legacy token, legacyTransactionRecords
      
         i++;
        } while(t>0);
        
        ownerId = uint64(ownerIdFromFlag(uint64(flag)));
      }

      // -------------------- compensate msg.sender ----------------------------
      
      {
        uint256 target = uint256(_mem[7+(ownerId*5)]) & k_addressMask;          // receiver of new spinoff address, new address of sending member

        if (msg.value<cost || payable(this).balance==0) return true;            // avoid underflow error
        uint payback = uint(msg.value)-cost;                                    // msg.value
        
        if (payable(this).balance < payback) payback = payable(this).balance;
        
        my_require(payable(address(uint160(target))).send(payback),"P");
        emit Deposit(address(uint160(target)),payback);
      }
      
      return true;
    }
    
    function confirmSpinOffGroup_L51b(bytes32[] calldata _in, bytes calldata data32, bytes32[] calldata _mem, bytes32[] calldata _abi) external payable {
        uint totalGas = gasleft();
        
        uint _tId = uint256(_in[0]);
        if (_tId>0) my_require(msg.value>0,"Y");
        uint256 t = tArr[_tId];
    
        uint64 f = getFlagsT(t);
        my_require(uint64(f&getOwnerMask(msg.sender))==0 && getAssetT(t)<128 && (((t & k_typeMask)>>252) == k_spinoffGroup) && isAddressOwner(msg.sender),"D");

        uint64 f_s = uint64( (uint256( uint256( tArr[checkSplitPreconditions(_tId,128)] ) & k_flagsMask )>>216) & k_flags2Mask );
        if ((f_s&getOwnerMask(msg.sender))==0) return;                                                   // ignore non-eligable votes 

        f  = uint64(f|getOwnerMask(msg.sender));                                                         // confirm spin-off transaction

        if (uint64(uint64(f>>5) & uint64(f_s>>5))==uint64(f_s>>5)) {                                     // executeMultiSigCommand(_tId,t,f) immediately
          
          getRsv().setABI(getHash(),128,abi.encodePacked(_abi));                                         // update current group - before executing spin-off
          
          AbstractGroupWalletFactory(GWF).replicate_group_l9Y{value: msg.value}(_in,data32,_mem);        // payout/return 100% = msg.value to new spinoff contract
          
          saveExecuted(_tId,f,t);                                                                        // spinoff transaction saved as executed and completed
          
          my_require(sendSpinoffAssets(_tId, f, _mem, totalGas),"U");                                    // pay back deposits, transfer legacy shares and contracts
          
        }
        else {
          saveNotExecuted(_tId,f,t);
        }
    }
    
    function drainLegacySharesGWP(bytes32 dhash,uint nb,address target) internal {
      uint i=0;
      uint256 tt;      
      do {
        tt = tArr[i];

        if ((tt & k_typeMask)>>252==k_legacyToken) {                            // cmd type = 11 = legacy token contract
          AbstractTokenProxy( address(uint160(uint(uint256(tt) & k_addressMask))) ).drainLegacyShares(dhash,owners[nb],target);
        }

        i++;
       } while(tt>0);
    }

    function submitFirstTransaction_gm(uint firstTRecord, uint256 dhash) external payable { 
      bool isLegacyToken = isLegacyTokenCmd(firstTRecord);
      my_require(tArr[0]==0&&(isAddressOwner(msg.sender) || isLegacyToken),"E"); // only first transaction, member transaction or legacyProxy contract
      
      uint               mask = k_flags3Mask;
      if (isLegacyToken) mask = k_flags4Mask;

      tArr[0] = uint256( (uint256( uint64( uint64(owners.length>>1)+1 ) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256( uint256(firstTRecord) & mask );
      
      tArr[uint256(uint160(GWF))] = dhash;      // GWP stores project domain hash
      
      emit GroupWalletDeployed(msg.sender,owners.length,uint256(block.timestamp));
    }
    
    function submitLegacyTransaction(uint tNb,uint tRecord) external payable {
      my_require(tNb>0 && tArr[tNb]==0 && isLegacyTokenCmd(tRecord),"H");
      tArr[tNb] = uint256(uint256( uint256(uint256( uint64( uint64(owners.length>>1)+1 ) )<<216) & k_flagsMask ) + uint256( uint256(tRecord) & k_flags4Mask )) | k_executeFlag;
    }

    function _submitTrans(uint _tRecord) internal {
      tArr[msg.value] = uint256( uint256( uint256( uint64( getTRequired(msg.value-1) ) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256( _tRecord & k_flags3Mask );
    }

    function __valueAndOwnerOK() private view {
      my_require(msg.value>0&&isAddressOwner(msg.sender)&&tArr[msg.value]==0,"u");
    }

    function submitTransaction_m79(uint aTRecord) public payable {
      __valueAndOwnerOK();
      _submitTrans(aTRecord);
    }

    function submitInvestTransaction_b3j(uint aTRecord) public payable { // 0x0000c7c1
      uint8 asset = getAssetT(aTRecord);
      __valueAndOwnerOK();
      my_require((asset==120||asset==118),"M");
      _submitTrans(aTRecord);
      executeMultiSigCommand(msg.value,tArr[msg.value],getFlags(msg.value));
    }

    function submitSplitTransaction(uint aTRecord, bytes32 _commitment, bytes32 _dname) external payable { 
      checkPreconditions(msg.value,aTRecord,k_splitGroup);
      my_require(checkSplitPreconditions(msg.value,0)==0,"l");

      _submitTrans(aTRecord);                                                                        // transaction type 7 will NOT be executed by majority vote
      
      tArr[uint256(_commitment)&k_addressMask]     = uint256(_commitment);                           // save 32-byte commitment
      tArr[(uint256(_commitment)&k_addressMask)+1] = uint256(_dname);                                // save spinOff / split project domain name
      
      AbstractGroupWalletFactory(GWF).reserve_replicate(dHashFromLabelBytes32(_dname), _commitment); // commitment = reserve new spinOff domain
    }
    
    // ENS V3 2023 requirements

    function isENSV3() internal view returns (bool) {
      return (address(AbstractGroupWalletFactory(GWF).ensNameWrapper()) != address(0x0));
    }

    function onERC1155Received(address,address,uint256,uint256,bytes calldata) public pure returns (bytes4 r) {
      return this.onERC1155Received.selector;
    }

    function setENSSubnode(bytes32 _label) internal {
      getENS().setSubnodeRecord(getHash(),_label,address(this),address(getRsv()),uint64(block.timestamp*1000) & k_timeMask); // e.g joe.ethereum.eth
    }

    function setENSV3Subnode(string memory _label) internal {
      getNWrap().setSubnodeRecord(getHash(),_label,address(this),address(getRsv()),uint64(block.timestamp*1000) & k_timeMask,0,0);
    }

    function getENSOwnerExpected(bytes32 _hash) private view returns (address) {
      if (isENSV3()) return AbstractGroupWalletFactory(GWF).getOwner(_hash);
      else           return getENS().owner(_hash);
    }

    function submitExecuteSplitTransaction(uint aTRecord, uint256 _secret) external payable {
      uint tNb = msg.value&0x000000000000000000000000000000000000000000000000000000ffffffffff;
      checkPreconditions(tNb,aTRecord,k_spinoffGroup);

      uint splitTNb = checkSplitPreconditions(tNb,0);
      my_require(uint256(tArr[uint160(aTRecord)&k_addressMask]&k_addressMask) == (uint256(aTRecord)&k_addressMask)&&splitTNb>0,"Z"); // check commitment lower 160 bits
      
      tArr[splitTNb] = uint256( uint256(tArr[splitTNb]) | k_executeFlag);                  // mark: split transaction type 7 = executed
      
      // -------------- register new domain ------------------------------------
      
      bytes32 splitName    = bytes32(tArr[ uint256(uint160( uint256( aTRecord ) & k_addressMask ) + 1) ]);
      bytes32 dHash        = dHashFromLabelBytes32(splitName);
      string memory _dname = labelStrFromBytes32(splitName);                               // domainName
      
      if (!isENSV3()) {
        getCtrl().registerWithConfig{value: getCtrl().rentPrice(_dname,31536000)}(_dname,address(this),31536000,bytes32(_secret),address(getRsv()),GWF);
        getRsv().setName(dHash,string(abi.encodePacked(_dname,AbstractGroupWalletFactory(GWF).tld())));
        getENS().setOwner(dHash,GWF);
      }
      else
      {
        (uint _rent,uint _dd) = AbsETHRegV3(address(getCtrl())).rentPrice(_dname,31536000);// rent ENS V3
        _dd = 0;                                                                           // silence compiler warning
        AbstractGroupWalletFactory(GWF)._register(dHash,_rent,_dname,31536000,bytes32(_secret));
      }
      
      tArr[tNb] = uint256( (uint256( uint64( getTRequired(tNb-1) ) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256( uint256(aTRecord) & k_flags3Mask );
    }

    function submitTransaction_addUser_kCY(uint256 aTRecord, uint256 dhash, uint256 label, uint256 len, uint256 bytesLen, bytes calldata data) external payable {
      my_require( (msg.value>0 || (msg.value==0 && dhash>0x0)) && isAddressOwner(msg.sender) && (((aTRecord & k_typeMask)>>252==k_addOwner)),"B");
  
      uint256 targetId     = uint256(aTRecord&k_addressMask);
      
      tArr[targetId+1]     = label;                                                // ENS2 = labelHash || ENS3 = label string with length byte
      tArr[targetId+2]     = len;                                                  // nb of bytes
      tArr[targetId+3]     = bytesLen;                                             // nb padding bytes
      structures[targetId] = data;                                                 // save new structure with added member
      
      if (dhash>0x0) {                                                             // is first first transaction nb = 0, store domain hash
        tArr[0]            = uint256( (uint256( uint64( uint64(owners.length>>1)+1 ) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256( uint256(aTRecord) & k_flags3Mask );
        tArr[uint256(uint160(GWF))] = dhash;                                       // store project domain hash
        emit GroupWalletDeployed(msg.sender,owners.length,uint256(block.timestamp));
      } else
      {
        _submitTrans(aTRecord);
      }
      
      my_require(getENSOwnerExpected(getHash())==address(this),"O");
    }
    
    function submitTransaction_proposal(uint256 aTRecord, bytes calldata data) external payable {
      my_require((aTRecord & k_typeMask)>>252 == k_proposal,"N");
      structures[ uint256(aTRecord&k_addressMask) ] = data;                     // save updated structure including claim,unique,skill,benefit strings
      submitTransaction_m79(aTRecord);
    }
  
    function executeTransaction_G1A(uint _tId) external payable {
      if (msg.value==0) return;
      uint256 t = tArr[_tId];
      uint64 f  = uint64( (uint256( uint256( t ) & k_flagsMask )>>216) & k_flags2Mask);
      my_require(uint64(f&getOwnerMask(msg.sender))>0&&getAssetT(t)<128,"I");
      
      if ( ( getFlags((msg.value-1)) & uint64(MAX_OWNER_COUNT) ) <= nbOfConfirmations(f) ) executeMultiSigCommand(_tId,t,f);
    }
    
    function getMemberWelcome(address target) external view returns (bytes memory) {
      my_require(isAddressOwner(target),"J");
      return abi.encode( tArr[uint256(uint160(target)+2)], tArr[uint256(uint160(target)+3)] );
    }
    
    function welcomeOneNewOwner(address target) internal {
      bytes32 l_dHash         = getHash();                                      // domainHash
      bytes32 l_label         = bytes32( tArr[uint256(uint160(target)+1)] );    // ens2 labelhash || ens3 labelString with length
      bytes32 l_dlabelHash;
      string memory l_labelStr;

      if (!isENSV3()) {
        l_dlabelHash = keccak256( abi.encodePacked(l_dHash,l_label) );
        setENSSubnode(l_label);                                                 // l_label = labelhash
      }
      else
      {
        l_labelStr   = labelStrFromBytes32(l_label);
        l_dlabelHash = getLabelDHash( keccak256(bytes(l_labelStr)) );
        setENSV3Subnode(l_labelStr);                                            // labelString
      }

      getRsv().setAddr(l_dlabelHash,target);                                    // set resolvedAddress to member addr
      getRsv().setABI (l_dHash,32,abi.encodePacked(structures[uint256(uint160(target))])); // update group structure, include new member
      
      if (!isENSV3()) getENS().setOwner(l_dlabelHash,target);
      else            getNWrap().setSubnodeOwner(l_dHash,l_labelStr,target,0,0);
      
      if (address(this).balance>welcomeBudget) {                                // gets e.g. 0.0025 ether minimal budget
        my_require(payable(address(uint160(target))).send(welcomeBudget),"F");
        emit Deposit(target, welcomeBudget);          
      }
      
       getPtoken(l_dHash).transfer_G8l(target, 10000);                          // 100 welcome shares
    }
    
    function executeMultiSigCommand(uint _tId,uint256 t,uint64 f) internal {
      uint    aValue;
      uint8   cmd    = uint8( (uint256( uint256( t ) & k_typeMask )>>252) & k_type2Mask );             // cmd type 0..15
      address target = address(uint160( uint256( t ) & k_addressMask ));

      if (cmd==2) {                                                                                    // addOwner
        my_require(!isAddressOwner(target)&&(owners.length<MAX_OWNER_COUNT)&&uint64(f & uint64(MAX_OWNER_COUNT)) >= 2,"2");
        owners.push(target);
        welcomeOneNewOwner(target);
        return saveExecuted(_tId,adjustMajority(f),t);
      }
      
      if (cmd==3) {                                                                                    // removeOwner OK with ENSV3 
        removeOneOwner(target);
        return saveExecuted(_tId,adjustMajority(f),t);
      }

      uint8 asset = getAssetT(t);                                                                      // input parameter e.g. to change requirement

      if (cmd==5) {                                                                                    // changeRequirement
        my_require((asset>=2) && (asset<=MAX_OWNER_COUNT),"5");
        return saveExecuted(_tId,(uint64(f|uint64(MAX_OWNER_COUNT)) ^ uint64(MAX_OWNER_COUNT))+uint64(asset),t);
      }

      if ((cmd==1)||(cmd==6)||(cmd==12)||(cmd==13)||(cmd==15)) 
        aValue = uint(uint256(t & k_valueMask)>>160) & k_value2Mask;                                   // nb of token/shares to be transferred, label of proposal


      if (cmd==1) {                                                                                    // send ETH
        if (asset==119) {                                                                              // ***** in case of a finalize auction transaction: target === BidBucket proxy contract
          address targGW = ABSReg(ABSExtDeed(target).registrar()).getGWPfromAuction( ABSExtDeed(target).lhash());
          getIntention().storeInvestment( AbstractTokenProxy(AbsGWM(targGW).proxyToken()).balanceOf( address(ABSExtDeed(target).registrar()) ), targGW, target.balance, getHash());
          tArr[_tId+1] = k_typeLegacyMsk + uint256( uint256( uint256( getTRequired(_tId) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256(uint160( AbsGWM(targGW).proxyToken() ) & k_addressMask) | k_executeFlag; // store proxyToken contract somewhere
        }

        (bool succ,bytes memory returnData) = target.call{ value: aValue<<20 }("");                    // mul 1.048.576 = 2**20 THis call to the BidBucket, e.g. , runs registrar.finalizeExternalAuction_WmS(lhash);
        require(succ,string(abi.encode( returnData, returnData.length )));

        emit Deposit(target, uint256(aValue<<20));
        return saveExecuted(_tId,f,t);
      }
    
      if (cmd==12) {                                                                                   // k_letterOfIntent
        uint l_rent = getCtrl().rentPrice("abcde",31536000);
        my_require(aValue!=0&&address(this).balance>=l_rent,"j");
        getIntention().saveLetterOfIntent{value:l_rent}(target,uint(aValue));                          // transaction payment required, in order to avoid spam and LoI grabber
      }

      AbstractTokenProxy l_token = AbstractTokenProxy(proxyToken());                                   // project ProxyToken contract

      if (cmd==6) {                                                                                    // transferShares / transfer GroupWallet Token
        my_require(aValue>0,"6");
        if (asset==125)                   l_token.sell_LA2(aValue);                                    // 125 = sell groupWallet shares to tokenContract to get liquidity
        if ((asset==126) || (asset==124)) l_token.transfer_G8l(target,aValue);                         // 126 = shares transaction | 124 = claim bonus shares

        if (asset<64) {
          AbstractTokenProxy(getTarget(uint(asset))).transfer_G8l(target,aValue);                      // <64 = transfer/convert investment group shares of other groups to target === Ungravel Global Shares
          AbsGlobalT(target).convertGroupSharesToEther(getTarget(uint(asset)),aValue);                 // call Ungravel Global Shares contract to convert group shares to ETH, apply for conversion and liquidity
        }
        
        return saveExecuted(_tId,f,t);
      }

      bytes32 dHash = getHash();                                                                       // domain hash of this group wallet name, s.a. "my-group.blast"

      if ((cmd==13)&&(aValue!=0x44bb0000000)) {                                                        // proposal to change pitch tags #
        getRsv().setABI( dHash,32,abi.encodePacked(structures[uint256(uint160(target))]) );            // proposal, except polls, update abi32 with group structure and pitch
      }
      
      if (cmd==14) {                                                                                   // upgrade
        if (asset==127) AbstractGroupWalletFactory(GWF).upgradeGWF(dHash,target);                      // upgrade GWF contract
        if (asset==123) l_token.upgradeTokenMaster(dHash,target);                                      // upgrade TMC contract
        if (asset==122) l_token.upgradeGWM(dHash,target);                                              // upgrade GWM contract
      }
      
      if (cmd==15) {                                                                                   // start funding auction
        uint period = (uint(uint64(asset) & 127) >> 2)*60;                                             // auction reveal period in minutes * 60 = seconds
        AbstractAuctionRegistrar(target).startAuction_ge0( keccak256( bytes(bytes32ToStr(bytes32(uint256(aValue)<<208))) ),period); // domain name = tag of auction
      
        aValue = uint(uint64(asset) & 3);
        l_token.transfer_G8l(target,uint64( (1+(aValue*aValue))*100*10000 ));                          // auction registrar holds assets = group shares
        
        if (!isENSV3()) setENSSubnode( getLabelHash(t) );                                              // assign ENS name, s.a. "seed.groupwallet.eth"
        else          setENSV3Subnode( bytes32ToStr(bytes32(uint256(t&k_valueMask)<<48)) );

        setResolvedAddress(t,address(this));
      }
    
      if ((cmd==7)||(cmd==8))  return saveNotExecuted(_tId,f,t);                                       // split-group | spin-off, DO NOT execute split-group yet
      
      //if (cmd==9)  return saveExecuted(_tId,f,t);                                                    // migrate group to another chain, x-chain migration
      //if (cmd==11) return saveExecuted(_tId,f,t);                                                    // legacy token, shares from another group before split-group
      
      return saveExecuted(_tId,f,t);
    }
    
    function TransferTokenFromContract(address _token, bytes memory data) public payable {             // requires a tokenAllow(address,address) in advance
      AbstractTokenProxy tProxy = AbstractTokenProxy( _token );                 // project ProxyToken contract

      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
    }
    
    function removeOneOwner(address target) internal {
      uint bal = address(this).balance;
      
      if (bal>lowBudget) {
        uint val = (bal-lowBudget) / activeOwners();
        if ((val>0) && (val<bal)) {
          my_require(payable(address(uint160(target))).send(val),"R");
          emit Deposit(target, val);          
        }
      }

      ownerChange(target, address(0x0)); // removeOwner
    }
    
    function activeOwners() private view returns (uint8 count) {
      count = 0;
      
      uint i=0;
      do {
        if (uint160(owners[i])!=0x0) count++; 
        i++;
      } while(i<owners.length);

      return count;
    }
    
    function getLabelHash(uint256 t) private pure returns (bytes32) {
      return keccak256(bytes(bytes32ToStr(bytes32(uint256(t&k_valueMask)<<48)))); // e.g. hash of "seed"
    }
    
    function getLabelDHash(bytes32 _lhash) private view returns (bytes32) {
      return keccak256(abi.encodePacked(getHash(),_lhash));
    }
    
    function setResolvedAddress(uint256 t, address target) private {
      getRsv().setAddr(getLabelDHash(getLabelHash(t)),target);
    }
    
    function reserveAuctionENSname(uint _tId, uint _payment, address deedContract) external {
      uint256 t = tArr[_tId];
      my_require(proxyToken()==msg.sender&&t>0&&(t&k_executeFlag)>0,"W");       // only pTokenContract
      
      setResolvedAddress(t,deedContract);                                       // setAddress of "seed.groupwallet.eth"

      my_require((payable(this).balance>(_payment*5)/1000)&&AbstractGroupWalletFactory(GWF).receivePayment{value: ((_payment*5)/1000)}(),"p"); // pay 0.5% ungravel license fee to GWF
    }
    
    function nameAuctionBidBucketLabel(bytes32 labelhashOrStr, address deedContract) external {
      string memory label;
      bytes32 l_dlabelHash;                                                     // labelhashOrStr = hash of label || label string with len
                                                                      
      my_require(proxyToken()==msg.sender,"T");                                 // only pTokenContract

      if (!isENSV3()) {
        setENSSubnode(labelhashOrStr);
        l_dlabelHash = getLabelDHash(labelhashOrStr);
      }
      else
      {
        label = labelStrFromBytes32(labelhashOrStr);
        setENSV3Subnode(label);                                                // assign ENS name, s.a. "bucket1.groupwallet.eth" || "bucket1"
        l_dlabelHash = getLabelDHash( keccak256(bytes(label)) );
      }

      getRsv().setAddr(l_dlabelHash,deedContract);
    }
    

    function getTNumberPublic() public view returns (uint count) {
      count = 0;
      
      if (tArr[0]==0) return count;
      
      uint i = 0;
      do {
        if (tArr[i] > 0) count += 1;
        i++;
      } while(tArr[i] > 0);
    }

    function isConfirmed(uint _tNb) public view returns (bool) {
      uint64 r;
      
      if (_tNb >0)           r = uint64(getTRequired(_tNb-1));
      if ((_tNb==0)||(r==0)) r = uint64(owners.length>>1)+1;

      return (r <= nbOfConfirmations(getFlags(_tNb)));
    }

    function getRequiredPublic(uint _tNb) external view returns (uint count) { 
      if (_tNb==0) return uint( uint64(owners.length>>1)+1 );
      else return getTRequired(_tNb-1);
    }
    
    function getIsOwner(address _owner) external view returns (bool) {
      return isAddressOwner(_owner);
    }
    
    function getTransactionsCount() external view returns (uint) {
      return getTNumberPublic();
    }
    
    function getTransactions(uint _tNb) external view returns (address destination, uint value, uint8 asset, bool executed, uint64 flags, uint8 typ, bool conf) {
      if (getTNumberPublic()>0)
        return (getTarget(_tNb),getTValue(_tNb),getAsset(_tNb),isTExecuted(_tNb),getFlags(_tNb),getType(_tNb),isConfirmed(_tNb));
    }
    
    function getTransactionRecord(uint _tNb) external view returns (uint256) {
      if (getTNumberPublic()>0) return tArr[_tNb];
      return 0;
    }
    
    function getAllTransactions() public view returns (uint256[] memory transArr) {
      if (tArr[0]==0) return new uint256[](0);
      
      uint count = 0;
      uint     i = 0;
      do {
        if (tArr[i] > 0) count += 1;
        i++;
      } while(tArr[i] > 0);
      
      uint256[] memory resultArr = new uint256[](count);

      i = 0;
      uint256 t;
      address token;
      do {
        t = tArr[i];
        
        if (isLegacyTokenCmd(t)) {
          token = address(uint160(uint256( t & k_addressMask )));
          resultArr[i] = uint256(t & k_value3Mask) + uint256(uint256( uint256(AbstractTokenProxy(token).balanceOf(tx.origin)) << 160 ) & k_valueMask); // legacy token balance of current member
        }
        else
        {
          resultArr[i] = t;                                                     // transaction record
        }
        
        i++;
      } while(i<count);

      return resultArr;
    }

    function getConfirmationCount(uint _tNb) external view returns (uint) {   
      return uint(nbOfConfirmations(getFlags(_tNb)));
    }
    
    function getTransactionCount(bool pending, bool executed) external view returns (uint count) {
      uint i = 0;
      uint t = getTNumberPublic();
      
      if (t==0) return 0;
      
      do {
        if (pending && !isTExecuted(i) || executed && isTExecuted(i)) count += 1;
        i++;
      } while(i<t);
    }

    function addressConfirmations(uint _tNb,address _owner) external view returns (bool) {
      return ownerConfirmed(_tNb,_owner);
    }

    function getOwners() external view returns (address[] memory) {
      return owners;
    }

    function getConfirmations(uint _tId) external view returns (address[] memory _confirmations) {   
      uint m = owners.length;
      address[] memory confirmationsTemp = new address[](m);
      uint count = 0;
      uint i=0;
      
      do
      {
        if (ownerConfirmed(_tId,owners[i])) confirmationsTemp[count++] = owners[i];
        i++;
      } while (i<m);
      
      _confirmations = new address[](count);
      
      i=0;
      do
      {
        _confirmations[i] = confirmationsTemp[i];
        i++;
      } while (i<count);      
    }
    
    function nbOfConfirmations(uint64 confirmFlags) internal view returns (uint64 nb) {
      uint64 m = 32;
      uint o   = owners.length;
      
      do
      {
        if ((confirmFlags & m) > 0) nb++;
        m = m*2;
        o--;
      } while (o>0);
      
      return nb;
    }
    
    function isAddressOwner(address _owner) private view returns (bool) {      
      uint m = owners.length;
      if (m==0) return false;
  
      uint i=0;
      do {
        if (owners[i]==_owner) return true;
        i++;
      } while(i<m&&i<MAX_OWNER_COUNT);

      return false;
    }
    
    function ownerChange( address _owner, address _newOwner) private {
      uint i=0;
      do {
        if (owners[i]==_owner) { owners[i] = _newOwner; return; } 
        i++;
      } while(i<owners.length&&i<MAX_OWNER_COUNT);

      my_require(false,"h");
    }
    
    function getOwnerMask(address _owner) private view returns (uint64 mask) {
      mask=32;
      uint i=0;
      uint len = owners.length;
      do {
        if (owners[i]==_owner) return mask;
        mask = mask*2;
        i++;
      } while(i<len&&i<MAX_OWNER_COUNT);

      return 0;
    }
    
    function ownerIdFromFlag(uint64 _ownerFlag) private pure returns (uint64 id) {
      id=1;
      uint64 i=0;
      do {
        if (_ownerFlag==id) return i;
        id = id*2;
        i++;
      } while(i<MAX_OWNER_COUNT);

      return 0;
    }

    function isTExecuted(uint _tNb) private view returns (bool) {
      return (getAsset(_tNb)>127);
    }

    function getOpenSplitTransactionNb(uint _tNb, uint8 executed) private view returns (uint idx) {
      idx = 0;
      
      uint256 t;
      uint64 f;
      
      if ((tArr[0]==0) || (_tNb<=0)) return 0;
      
      uint i = _tNb;
      do {
        t = tArr[i];
        f = getFlagsT(t);
        
        if ( (t!=0) && (uint8(uint256((uint256( uint256( t ) & k_typeMask )>>252) & k_type2Mask)) == k_splitGroup) && ((getAssetT(t)&128)==executed) && (nbOfConfirmations(f)>0) ) return i;
        i--;
      } while(i>0);
      
      return 0;
    }

    function ownerConfirmed(uint _tNb, address _owner) private view returns (bool) {
      uint64 f = getFlags(_tNb);
      uint64 o = getOwnerMask(_owner);
      return (uint64(f&o)>0);
    }
    
    function getTRequired(uint _tId) private view returns (uint64) {
      return uint64(getFlags(_tId) & uint64(MAX_OWNER_COUNT));
    }
    
    function newProxyGroupWallet_j5O(address[] calldata _owners) external payable {
      uint i=0;
      do {
        my_require(_owners[i] != address(0x0) && owners.length==i,"a");
        owners.push(_owners[i]);
        i++;
      } while(i<_owners.length);      
      
      GWF = msg.sender;
    }
    
    function saveAsset(bytes32 dhash, uint256 asset, string calldata key, string calldata data) external payable {
      my_require(isAddressOwner(msg.sender)&&dhash!=0x0&&getENS().recordExists(dhash)&&getENSOwnerExpected(dhash)==address(this),"n");
      
      if (asset!=3) {
        uint256 pHash = uint256(getHash());
        my_require(pHash>0&&getENSOwnerExpected(bytes32(pHash))==address(this)&&bytes32(pHash)==dhash,"q");
      }

      getRsv().setText(dhash,key,data);
        
      if (asset==1) emit ColorTableSaved (dhash);
      if (asset==2) emit EtherScriptSaved(dhash,key);
    }

    function forwardEther(address payable receiver) external payable {
      my_require(msg.value>0&&receiver.send(msg.value),"f");
      emit Deposit(receiver, msg.value);
    }
    
    function depositEther(uint aTRecord, uint tNb) external payable {
      my_require(msg.value>0&&tNb>0&&tArr[tNb]==0&&uint256(uint256(aTRecord & k_typeMask)>>252)==10,"i");
      tArr[tNb] = uint256( uint256(uint256( uint64( getTRequired(tNb-1) ) | getOwnerMask(msg.sender) )<<216) & k_flagsMask ) + uint256( uint256(aTRecord) & (k_address2Mask-k_flagsMask) ) + uint256( msg.value & k_addressMask );
      emit Deposit(address(payable(this)), uint256(msg.value));
    }

    function setIntentionsContract(address intCtr) external payable {
      my_require(address(msg.sender)==masterCopy,"k");
      intent = intCtr;
    }

    function proxyToken() public payable returns (address) {
      return address(getPtoken(getHash()));
    } 

    fallback() external payable {
      if (msg.value > 0) {
        emit Deposit(address(this), msg.value);
        return;
      }
      my_require(false,"b");
    }
    
    receive() external payable { emit Deposit(msg.sender, msg.value); }

    function version() external pure returns(uint256 v) {
      return 20010090;
    }
    
    constructor(address[] memory _owners) payable {
      uint i=0;
      do {
        my_require(_owners[i] != address(0x0),"C");
        owners.push(_owners[i]);
        i++;
      } while(i<_owners.length);
      
      masterCopy  = address(msg.sender);                                        // save owner of GroupWalletMaster
    }
    
    uint constant     private MAX_OWNER_COUNT = 31;
    uint constant     private welcomeBudget   = 0.0025 ether;                   // new member gets welcome ether
    uint constant     private lowBudget       = 0.200 ether;                    // GWP - GroupWalletProxy contract keeps ether if members depart

    uint256 constant k_proposal       = 13;                                     // cmd type of the proposal     transaction type
    uint256 constant k_legacyToken    = 11;                                     // cmd type of the legacyToken  transaction type
    uint256 constant k_letterOfIntent = 12;                                     // cmd type of the legacyToken  transaction type
    uint256 constant k_deposit        = 10;                                     // cmd type of the deposit      transaction type
    uint256 constant k_splitGroup     =  7;                                     // cmd type of the splitGroup   transaction type
    uint256 constant k_spinoffGroup   =  8;                                     // cmd type of the spinoffGroup transaction type
    uint256 constant k_addOwner       =  2;                                     // cmd type of the addOwner     transaction type
    
    uint64  constant k_timeMask     = 0xffffffffffff0000;
    uint256 constant k_addressMask  = 0x000000000000000000000000ffffffffffffffffffffffffffffffffffffffff; // 160 bits
    uint256 constant k_address2Mask = 0xffffffffffffffffffffffff0000000000000000000000000000000000000000;
        
    uint256 constant k_valueMask    = 0x000000000000ffffffffffff0000000000000000000000000000000000000000;
    uint256 constant k_value3Mask   = 0xffffffffffff000000000000ffffffffffffffffffffffffffffffffffffffff;
    uint256 constant k_value2Mask   = 0x0000000000000000000000000000000000000000000000000000ffffffffffff;

    uint256 constant k_flagsMask    = 0x0fffffffff000000000000000000000000000000000000000000000000000000;
    uint256 constant k_flags2Mask   = 0x0000000000000000000000000000000000000000000000000000000fffffffff;
    uint256 constant k_flags3Mask   = 0xf000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffff;
    uint256 constant k_flags4Mask   = 0xffffffffe0ffffffffffffffffffffffffffffffffffffffffffffffffffffff;
    
    uint256 constant k_assetMask    = 0x0000000000ff0000000000000000000000000000000000000000000000000000; // only 8 bits
    uint256 constant k_asset2Mask   = 0x00000000000000000000000000000000000000000000000000000000000000ff;
    uint256 constant k_asset3Mask   = 0xffffffffff00ffffffffffffffffffffffffffffffffffffffffffffffffffff;
    uint256 constant k_executeFlag  = 0x0000000000800000000000000000000000000000000000000000000000000000; // 1 bit
    
    uint256 constant k_typeMask     = 0xf000000000000000000000000000000000000000000000000000000000000000; // command type = 4 bits
    uint256 constant k_type2Mask    = 0x000000000000000000000000000000000000000000000000000000000000000f;

    uint256 constant k_typeLegacyMsk= 0xb000000000000000000000000000000000000000000000000000000000000000; // legacy token contract
}