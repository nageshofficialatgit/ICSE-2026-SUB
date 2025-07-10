// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.18 <0.8.20;

// ungravel.eth, GroupWalletFactory, GroupWalletMaster, GroupWallet, ProxyWallet, TokenMaster, ProxyToken, PrePaidContract, AuctionMaster, BiddingProxy, intDeedMaster, extDeedMaster, IntDeedProxy, Invitations by pepihasenfuss.eth 2017-2025, Copyright (c) 2025

// GroupWallet and Ungravel is entirely based on Ethereum Name Service, "ENS", the domain name registry.

//   ENS, ENSRegistryWithFallback, PublicResolver, Resolver, FIFS-Registrar, Registrar, AuctionRegistrar, BaseRegistrar, ReverseRegistrar, DefaultReverseResolver, ETHRegistrarController,
//   PriceOracle, SimplePriceOracle, StablePriceOracle, ENSMigrationSubdomainRegistrar, CustomRegistrar, Root, RegistrarMigration are contracts of "ENS", by Nick Johnson and team.
//
//   Copyright (c) 2018, True Names Limited / ENS Labs Limited
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

interface ENS_F {
  event NewOwner(bytes32 indexed node, bytes32 indexed label, address owner);
  event Transfer(bytes32 indexed node, address owner);
  event NewResolver(bytes32 indexed node, address resolver);
  event NewTTL(bytes32 indexed node, uint64 ttl);
  event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

  function setRecord(bytes32 node, address owner, address resolver, uint64 ttl) external;
  function setSubnodeRecord(bytes32 node, bytes32 label, address owner, address resolver, uint64 ttl) external;
  function setSubnodeOwner(bytes32 node, bytes32 label, address owner) external returns(bytes32);
  function setResolver(bytes32 node, address resolver) external;
  function setOwner(bytes32 node, address owner) external;
  function setTTL(bytes32 node, uint64 ttl) external;

  function owner(bytes32 node) external view returns (address);
  function resolver(bytes32 node) external view returns (address);
  function ttl(bytes32 node) external view returns (uint64);
  function recordExists(bytes32 node) external view returns (bool);
  function isApprovedForAll(address ensowner, address operator) external view returns (bool);
}

abstract contract AbstractGWF_ENS {
  event NewOwner(bytes32 indexed node, bytes32 indexed label, address owner);
  event Transfer(bytes32 indexed node, address owner);
  event NewTTL(bytes32 indexed node, uint64 ttl);
  event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

  function setRecord(bytes32 node, address owner, address resolver, uint64 ttl) external virtual;
  function setSubnodeRecord(bytes32 node, bytes32 label, address owner, address resolver, uint64 ttl) external virtual;
  function setSubnodeOwner(bytes32 node, bytes32 label, address owner) external virtual returns(bytes32);
  function setResolver(bytes32 node, address resolver) external virtual;
  function setOwner(bytes32 node, address owner) external virtual;
  
  function owner(bytes32 node) public view virtual returns (address);
  function recordExists(bytes32 node) external virtual view returns (bool);
  function isApprovedForAll(address ensowner, address operator) external virtual view returns (bool);
}

abstract contract AbstractGWF_ReverseRegistrar {
  function claim(address owner) external virtual returns (bytes32);
  function claimWithResolver(address owner, address resolver) external virtual returns (bytes32);
  function setName(string memory name) external virtual returns (bytes32);
  function node(address addr) external virtual pure returns (bytes32);
}

abstract contract AbstractBaseRegistrar {
  event NameMigrated(uint256 indexed id, address indexed owner, uint expires);
  event NameRegistered(uint256 indexed id, address indexed owner, uint expires);
  event NameRenewed(uint256 indexed id, uint expires);

  bytes32 public baseNode;   // The namehash of the TLD this registrar owns eg, (.)eth
  ENS_F public ens;
}

abstract contract AbsIntentionsGWF {
  function saveLetterOfIntent(address target, uint nbOfShares) public virtual payable;
  function hashOfGWP(AbstractGWF_GWP _gwp) internal virtual view returns (bytes32);
  function getIntendedLOIShares(address target, address investor) public virtual view returns (uint);
}

abstract contract AbstractGWF_GWP {
  function getGWF() external view virtual returns (address);
  function getIsOwner(address _owner) external virtual view returns (bool);
  function getOwners()                external virtual view returns (address[] memory);
  function newProxyGroupWallet_j5O(address[] calldata _owners) external virtual payable;
  function reverseENS(string calldata _domain, address _reverse) external virtual;
  function getTransactionsCount() external view virtual returns (uint);
  function getTransactionRecord(uint _tNb) external view virtual returns (uint256);
  function getIntention() public virtual view returns (AbsIntentionsGWF);
}

interface Abstract_TokenProxy {
  function newToken(uint256[] calldata _data) external payable;
}

abstract contract AbstractETHRegController {
  mapping(bytes32=>uint) public commitments;

  uint public minCommitmentAge;
  uint public maxCommitmentAge;

  address public nameWrapper;

  event NameRegistered(string name, bytes32 indexed label, address indexed owner, uint cost, uint expires);
  event NameRenewed(string name, bytes32 indexed label, uint cost, uint expires);
  event NewPriceOracle(address indexed oracle);

  function rentPrice(string memory name, uint duration) view external virtual returns(uint);
  function makeCommitmentWithConfig(string memory name, address owner, bytes32 secret, address resolver, address addr) pure external virtual returns(bytes32);
  function commit(bytes32 commitment) external virtual;
  function register(string calldata name, address owner, uint duration, bytes32 secret) external virtual payable;
  function registerWithConfig(string memory name, address owner, uint duration, bytes32 secret, address resolver, address addr) external virtual payable;
  function available(string memory name) external virtual view returns(bool);
  function register(string calldata name,address owner,uint256 duration,bytes32 secret,address resolver,bytes[] calldata data,bool reverseRecord,uint16 ownerControlledFuses) external virtual payable;
}

abstract contract AbstractGWF_Resolver {
  mapping(bytes32=>bytes) hashes;

  event AddrChanged(bytes32 indexed node, address a);
  event AddressChanged(bytes32 indexed node, uint coinType, bytes newAddress);
  event NameChanged(bytes32 indexed node, string name);
  event ABIChanged(bytes32 indexed node, uint256 indexed contentType);
  event PubkeyChanged(bytes32 indexed node, bytes32 x, bytes32 y);
  event TextChanged(bytes32 indexed node, string indexed indexedKey, string key);
  event ContenthashChanged(bytes32 indexed node, bytes hash);
  
  function ABI(bytes32 node, uint256 contentTypes) external virtual view returns (uint256, bytes memory);
  function addr(bytes32 node) external virtual view returns (address);
  function addr(bytes32 node, uint coinType) external virtual view returns(bytes memory);
  function name(bytes32 node) external virtual view returns (string memory);
  function text(bytes32 node, string calldata key) external virtual view returns (string memory);
  function supportsInterface(bytes4 interfaceId) external virtual view returns (bool);
  function setApprovalForAll(address operator, bool approved) virtual external;

  function setABI(bytes32 node, uint256 contentType, bytes calldata data) external virtual;
  function setAddr(bytes32 node, address r_addr) external virtual;
  function setAddr(bytes32 node, uint coinType, bytes calldata a) external virtual;
  function setName(bytes32 node, string calldata _name) external virtual;
  function setText(bytes32 node, string calldata key, string calldata value) external virtual;
}

abstract contract Abstract_GWF {
  AbstractGWF_Resolver            public  resolverContract;
  AbstractETHRegController        public  controllerContract;
  AbstractBaseRegistrar           public  base;
  AbstractGWF_ENS                 public  ens;
  AbstractGWF_ReverseRegistrar    public  reverseContract;
  address                         public  GWFowner;
  
  mapping(uint64=>uint256)        private installations;                        // installTime +  proxyTokenAddr
  mapping(bytes32=>uint256)       private commitments;                          // commitment  +  ownerAddr
  
  function version() public pure virtual returns(uint256 v);
  function getOwner(bytes32 _domainHash) external virtual view returns (address);
  function importGWP(bytes32 _dHash, uint256 commitment, uint256 installation) external virtual payable;
  function getGWProxy(bytes32 _dHash) public virtual view returns (address);
  function getProxyToken(bytes32 _domainHash) public virtual view returns (address p);
}

interface Abstract_GWPC {
  function getMasterCopy() external view returns (address);
}

abstract contract Abs_AuctionRegistrar {
  function startAuction_ge0(bytes32 _hash, uint revealP) public virtual payable;
}

abstract contract NWrapper {
  function setSubnodeRecord(bytes32 parentNode,string memory label,address owner,address resolver,uint64 ttl,uint32 fuses,uint64 expiry) external virtual returns (bytes32 node);
  function setSubnodeOwner(bytes32 node,string calldata label,address newOwner,uint32 fuses,uint64 expiry) external virtual returns (bytes32);
  function setApprovalForAll(address operator,bool approved) external virtual;
}

interface I_NWrapper {
  function setSubnodeRecord(bytes32 parentNode,string memory label,address owner,address resolver,uint64 ttl,uint32 fuses,uint64 expiry) external returns (bytes32 node);
  function setSubnodeOwner(bytes32 node,string calldata label,address newOwner,uint32 fuses,uint64 expiry) external returns (bytes32);
  function setApprovalForAll(address operator,bool approved) external;
}


/// @title Proxy - Generic proxy contract allows to execute all transactions applying the code of a master contract.
/// @author Stefan George - <stefan@gnosis.pm> /// ProxyToken adapted and applied for shares and token by pepihasenfuss.eth
pragma solidity ^0.8.18 <0.8.20;

contract ProxyToken {
    address internal masterCopy;

    bytes32 internal name32;
    uint256 private ownerPrices;

    mapping (address => uint256) private balances;
    mapping (address => mapping  (address => uint256)) private allowed;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event FrozenFunds(address target, bool frozen);
    event Deposit(address from, uint256 value);
    event Deployment(address owner, address theContract);
    event Approval(address indexed owner,address indexed spender,uint256 value);

    constructor(address _masterCopy) payable
    {
      masterCopy = _masterCopy;
    }
    
    fallback () external payable
    {   
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            let master := and(sload(0), 0xffffffffffffffffffffffffffffffffffffffff)
            if eq(calldataload(0), 0xa619486e00000000000000000000000000000000000000000000000000000000) {
                mstore(0, master)
                return(0, 0x20)
            }

            let ptr := mload(0x40)
            calldatacopy(ptr, 0, calldatasize())
            let success := delegatecall(gas(), master, ptr, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            if eq(success, 0) { 
              if eq(returndatasize(),0) { revert(0, 0x404) }
              revert(0, returndatasize())
            }
            return(0, returndatasize())
        }
    }
    
    function upgrade(address master) external payable {
      require(master!=address(0x0)&&masterCopy==msg.sender,"pt!");
      masterCopy = master;
    }
    
    receive() external payable { emit Deposit(msg.sender, msg.value); }         // *** PTC may receive payments ***
}

/// @title Proxy - Generic proxy contract allows to execute all transactions applying the code of a master contract.
/// @author Stefan George - <stefan@gnosis.pm> /// ProxyGroupWallet adapted and applied for GroupWallet by pepihasenfuss.eth
pragma solidity ^0.8.18 <0.8.20;

contract ProxyGroupWallet {
    address internal masterCopy;

    mapping(uint256 => uint256) private tArr;
    address[]                   private owners;
    
    address internal GWF;                                                       // GWF - GroupWalletFactory contract
    mapping(uint256 => bytes)   private structures;
  
    // *************************************************************************
    event Deposit(address dep_from, uint256 dep_value);
    
    constructor(address _masterCopy, string memory _domain, AbstractGWF_ReverseRegistrar _reverse) payable
    { 
      masterCopy = _masterCopy;
      _reverse.setName(_domain);
    }
    
    fallback () external payable
    {   
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            let master := and(sload(0), 0xffffffffffffffffffffffffffffffffffffffff)
            if eq(calldataload(0), 0xa619486e00000000000000000000000000000000000000000000000000000000) {
                mstore(0, master)
                return(0, 0x20)
            }

            let ptr := mload(0x40)
            calldatacopy(ptr, 0, calldatasize())
            let success := delegatecall(gas(), master, ptr, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            if eq(success, 0) { 
              if eq(returndatasize(),0) { revert(0, 0x504) }
              revert(0, returndatasize())
            }
            return(0, returndatasize())
        }
    }
    
    function upgrade(address master) external payable {
      bytes32 hash = bytes32(tArr[uint256(uint160(GWF))]);
      address gwp  = Abstract_GWF(GWF).getProxyToken(hash);
      require(master!=address(0x0)&&msg.sender==Abstract_GWPC(gwp).getMasterCopy(),"gwp!");
      masterCopy = master;
    }
    
    receive() external payable { emit Deposit(msg.sender, msg.value); }         // *** GWP can sell common shares to TokenProxy, thus receiving payments ***
}

// GroupWalletFactory2 by pepihasenfuss.eth 2017-2023
contract GroupWalletFactory2 {

    event TestReturn(uint256 v1, uint256 v2, uint256 v3, uint256 v4);
    event Deposit(address from, uint256 value);
    event PrePaidAccountReady(address from, uint256 value);
    event StructureDeployed(bytes32 domainHash);
    event ColorTableSaved(bytes32 domainHash);
    event EtherScriptSaved(bytes32 domainHash,string key);
    event ProxyTokenCreation(ProxyToken proxy);
    event ProxyGroupWalletCreation(ProxyGroupWallet proxy);
    event SetPrices(bytes32 domainHash);
    event TransferOwner(bytes32 domainHash);
    event FreezeToken(bytes32 domainHash);
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    uint256 constant k_aMask        = 0x000000000000000000000000ffffffffffffffffffffffffffffffffffffffff;
    uint256 constant k_commitMask   = 0xffffffffffffffffffffffff0000000000000000000000000000000000000000;
    uint256 constant k_commit2Mask  = 0x0000000000000000000000000000000000000000ffffffffffffffffffffffff;
    uint256 constant k_lockedMask   = 0x0000000000000000000000010000000000000000000000000000000000000000;
    
    bytes32 constant k_abi80        = 0x0000000000000000000000000000000000000000000000000000000000000080;
    uint256 constant k_rentMask     = 0x00000000000000000000000000000000ffffffffffffffffffffffffffffffff;

    address constant k_add00        = address(0x0);


    AbstractGWF_Resolver            public  resolverContract;
    AbstractETHRegController        public  controllerContract;
    AbstractBaseRegistrar           public  base;
    AbstractGWF_ENS                 public  ens;
    AbstractGWF_ReverseRegistrar    public  reverseContract;
    address                         public  GWFowner;
    NWrapper                        public  ensNameWrapper;
    Abs_AuctionRegistrar            public  auctionContract;
    
    mapping(uint64=>uint256)        private installations;                      // installTime +  proxyTokenAddr
    mapping(bytes32=>uint256)       private commitments;                        // commitment  +  ownerAddr

  
    modifier onlyByOwner() {
      require(GWFowner==msg.sender,"oO!");
      _;
    }

    modifier calledByUngravelGWP() {                                                                // caller MUST BE valid contract, a GroupWalletProxy GWP belonging to Ungravel
      require(address(msg.sender)!=k_add00&&isContract(msg.sender),"UG");                           // GWP uses GWF contract
      bytes32 hsh  = getdHash(AbstractGWF_GWP(msg.sender));                                         // *** hsh of seed.eth ***  hsh = GroupWallet domain hash 
      address gwfc = AbstractGWF_GWP(msg.sender).getGWF();                                          // GWF contract, derived from GWP, in this case the caller
      require(gwfc==address(this)&&hsh!=0x0&&Abstract_GWF(gwfc).getOwner(hsh)==msg.sender,"UG2");   // the requested GWP owns its own dName, s.a. "silvias-bakery.eth"
      _;
    }

    modifier onlyGWF(address _ctr) {                                                                // called by upgradeGWF() & importGWP()
      require(GWFowner!=address(0x0)&&GWFowner==Abstract_GWF(_ctr).GWFowner(),"gf!");
      _;
    }

    modifier calledByUNG_GWP(bytes32 hash) {                                                        // caller MUST BE valid contract, a GroupWalletProxy GWP, belonging to Ungravel
      address gw = Abstract_GWF(msg.sender).getOwner(hash);
      require(hash!=0x0&&gw!=address(0x0)&&AbstractGWF_GWP(gw).getIsOwner(tx.origin),"UG3");        // origin is member of the GWP
      _;
    }

    function isContract(address addr) internal view returns (bool) {
      uint size;
      assembly { size := extcodesize(addr) }
      return size > 0;
    }

    function getdHash(AbstractGWF_GWP gw) internal view returns (bytes32) {
      return bytes32(gw.getTransactionRecord(uint256(uint160(gw.getGWF()))));   // domain hash, s.a. hash of "mygroupwallet.eth"
    }

    function getGWF() external view returns (address) {
      return address(this);                                                     // needed, to call GroupWallet or GroupWalletFactory transparently
    }

    function getCommitment(bytes32 _domainHash) private view returns (uint64 comm) {
      return uint64( (uint256( commitments[_domainHash] & k_commitMask )>>160) & k_commit2Mask );
    }
    
    function getOwner(bytes32 _domainHash) external view returns (address) {
      return address( uint160( commitments[_domainHash] & k_aMask ) );
    }
    
    function saveOwner(address _iToken, bytes32 _domainHash) private {
      commitments[ _domainHash ] = uint256(uint160(_iToken)) + uint256( commitments[_domainHash] & k_commitMask);
    }

    function saveCommitment(bytes32 input, bytes32 _domainHash) private {
      commitments[_domainHash] = uint256( (uint256(input)<<160) & k_commitMask ) + uint256( commitments[_domainHash] & k_aMask);
    }

    function getInstallTime(bytes32 _domainHash) public view returns (uint256 iTime) {
      uint256 i = uint256(installations[ getCommitment(_domainHash) ]);
      return uint256( (uint256( uint256(i) & k_commitMask )>>160) & k_commit2Mask );
    }

    function getProxyToken(bytes32 _domainHash) public view returns (address p) {
      return address( uint160( uint256( uint256(installations[ getCommitment(_domainHash) ]) ) & k_aMask ) );
    }

    function getGWProxy(bytes32 _dHash) public view returns (address) {
      return address( uint160( commitments[_dHash] & k_aMask ) );
    }

    function saveProxyToken(address _iOwner, bytes32 _domainHash) private {
      uint64 hsh  = getCommitment(_domainHash);
      uint256 i = uint256(installations[ hsh ]);
      installations[ hsh ] = uint256(uint160(_iOwner)) + uint256(i & k_commitMask);
    }

    function saveInstallTime(uint256 input, bytes32 _domainHash) private {
      uint64 hsh  = getCommitment(_domainHash);
      uint256 i = uint256(installations[ hsh ]);
      installations[ hsh ] = uint256( (uint256(input)<<160) & k_commitMask ) + uint256(i & k_aMask);
    }

    // -------------------  owners ---------------------------------------------
    
    function getIsOwner(bytes32 _dHash,address _owner) external view returns (bool) {
      uint256 c = commitments[_dHash];
      address theGWPcontract = address( uint160( c&k_aMask ) );
      if (theGWPcontract==msg.sender) return false;                             // is initiator calling, no owners list yet
      return AbstractGWF_GWP( theGWPcontract ).getIsOwner(_owner);
    }

    function getOwners(bytes32 _dHash) external view returns (address[] memory) {
      uint256 c = commitments[_dHash];
      return getOwners_internal(c);
    }
    
    function getOwners_internal(uint256 c) private view returns (address[] memory) {
      address a = address( uint160( c&k_aMask ) );
      if (a!=tx.origin) return AbstractGWF_GWP( a ).getOwners();
      
      address[] memory empty;
      return empty;
    }
  
    // -------------------  strings ---------------------------------------------

    function char(bytes1 b) private pure returns (bytes1 c) {
        if (uint8(b) < uint8(10)) return bytes1(uint8(b) + 0x30);
        else return bytes1(uint8(b) + 0x57);
    }
    
    function strlen(string memory s) internal pure returns (uint) {
        uint len;
        uint i = 0;
        uint bytelength = bytes(s).length;
        for(len = 0; i < bytelength; len++) {
            bytes1 b = bytes(s)[i];
            if(b < 0x80) {
                i += 1;
            } else if (b < 0xE0) {
                i += 2;
            } else if (b < 0xF0) {
                i += 3;
            } else if (b < 0xF8) {
                i += 4;
            } else if (b < 0xFC) {
                i += 5;
            } else {
                i += 6;
            }
        }
        return len;
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
    
    function concatString(bytes32 _b, string memory _str) internal pure returns (string memory) {
      bytes memory bArr = new bytes(32);
      uint8 i = 0;
      uint8 off = 0;
      
      do
       { 
        if (_b[i]!=0) { 
          bArr[i] = _b[i];
        }
        else
        {
          off = i;
        }
        i++;
      } while(off==0&&i<32);
      
      i = 0;
      uint len = strlen(_str);
      
      do
       { 
        bArr[off+i] = bytes(_str)[i];
        i++;
      } while(i<len&&i<32);
      
      return string(bArr); 
    }

    function stringMemoryTobytes32(string memory _data) private pure returns(bytes32 a) {
      // solium-disable-next-line security/no-inline-assembly
      assembly {
          a := mload(add(_data, 32))
      }
    }
    
    function mb32(bytes memory _data) private pure returns(bytes32 a) {
      // solium-disable-next-line security/no-inline-assembly
      assembly {
          a := mload(add(_data, 32))
      }
    }
    
    function keccak(bytes memory self, uint offset, uint len) internal pure returns (bytes32 ret) {
        my_require(offset + len <= self.length,"k");
        
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            ret := keccak256(add(add(self, 32), offset), len)
        }
    }
  
    function memcpy(uint dest, uint src, uint len) private pure {
        // Copy word-length chunks while possible
        for (; len >= 32; len -= 32) {
            // solium-disable-next-line security/no-inline-assembly
            assembly {
                mstore(dest, mload(src))
            }
            dest += 32;
            src += 32;
        }
        
        if (len==0) return;

        // Copy remaining bytes
        uint mask = 256 ** (32 - len) - 1;
        
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            let srcpart := and(mload(src), not(mask))
            let destpart := and(mload(dest), mask)
            mstore(dest, or(destpart, srcpart))
        }
    }

    function substring(bytes memory self, uint offset, uint len) internal pure returns(bytes memory) {
        my_require(offset + len <= self.length,"s");

        bytes memory ret = new bytes(len);
        uint dest;
        uint src;

        // solium-disable-next-line security/no-inline-assembly
        assembly {
            dest := add(ret, 32)
            src  := add(add(self, 32), offset)
        }
        memcpy(dest, src, len);

        return ret;
    }
    
    function readBytes32(bytes memory self, uint idx) internal pure returns (bytes32 ret) {
        my_require(idx + 32 <= self.length,"b");
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            ret := mload(add(add(self, 32), idx))
        }
    }
    
    function bytes32ToAsciiString(bytes32 _bytes32, uint len) private pure returns (string memory) {
        bytes memory s = new bytes((len*2)+2);
        s[0] = 0x30;
        s[1] = 0x78;
      
        for (uint i = 0; i < len; i++) {
            bytes1 b = bytes1(uint8(uint(_bytes32) / (2 ** (8 * ((len-1) - i)))));
            bytes1 hi = bytes1(uint8(b) / 16);
            bytes1 lo = bytes1(uint8(b) - 16 * uint8(hi));
            s[2+(2 * i)] = char(hi);
            s[2+(2 * i) + 1] = char(lo);
        }
        return string(s);
    }

    function labelStrFromBytes32(bytes32 _label) private pure returns (string memory) {
      return bytesToStr(_label,uint(_label)&0xff);
    }

    function reportValue(uint256 a, string memory str) internal pure returns (string memory r) {
      return string(abi.encodePacked( bytes32ToAsciiString(bytes32(uint256( a )),32), str));
    }

    function getDomainHash(string memory _d) internal view returns (bytes32) {
      return keccak256(abi.encodePacked(base.baseNode(), keccak256(bytes(_d))));
    }
    
    // -------------------  GWF ---------------------------------------------

    function my_require(bool b, string memory str) private pure {
      require(b,str);
    }
    
    function reserve_ogn(bytes32 _domainHash,bytes32 _commitment,bytes calldata data) external payable {
      (bool success, bytes memory returnData) = address(0xDadaDadadadadadaDaDadAdaDADAdadAdADaDADA).call{value: 0}(data);
      
      my_require(data.length>0 && success && returnData.length==0 && _commitment!=0x0,"O");
      emit StructureDeployed(_domainHash);
      
      controllerContract.commit(_commitment);
      commitments[ _domainHash ] = uint256( (uint256(_commitment)<<160) & k_commitMask ) + uint256( uint160(msg.sender) & k_aMask ); // saveOwner initiator = owner
    }

    function reserve_replicate(bytes32 _domainHash,bytes32 _commitment) external payable {
      controllerContract.commit(_commitment);
      commitments[ _domainHash ] = uint256( (uint256(_commitment)<<160) & k_commitMask ) + uint256( uint160(msg.sender) & k_aMask ); // saveOwner initiator = owner
    }

    // ENS V3 2023 methods

    function isENSV3() internal view returns (bool) {
      return (address(ensNameWrapper)!=k_add00);
    }

    function onERC1155Received(address,address,uint256,uint256,bytes calldata) public pure returns (bytes4 r) {
      return this.onERC1155Received.selector;
    }

    function _allowNameWrapperAndResolver(address _owner) private {
      I_NWrapper(controllerContract.nameWrapper()).setApprovalForAll(_owner,true); // GWP gets approved for NameWrapper
      resolverContract.setApprovalForAll(_owner,true);                             // GWP gets approved for PublicResolver
    }

    function _register(bytes32 _dHash,uint256 _rent,string memory _name,uint256 _dur,bytes32 _secret) public payable {
      
      address l_ProjectOwner = address(uint160( commitments[_dHash] & k_aMask ));
      my_require((l_ProjectOwner==msg.sender || msg.sender==address(this))&&controllerContract.available(_name)&&_secret!=0&&_dur!=0,"Y");

     if (isENSV3()) {                                                          // register ENS V3
        bytes[] memory cmd = new bytes[](1);
        cmd[0] = abi.encodePacked( bytes4(0x8b95dd71), _dHash, uint256(0x3c), uint256(0x60), uint256(0x14), uint256(uint256(uint160(address(this)))<<96) );

        (bool success, bytes memory rdata) = address(controllerContract).call{value:_rent}
        ( abi.encodeWithSignature("register(string,address,uint256,bytes32,address,bytes[],bool,uint16)",_name,address(this),_dur,_secret,address(resolverContract),cmd,false,0) );

        my_require(success&&rdata.length==0,string(rdata));

        if (l_ProjectOwner==msg.sender) _allowNameWrapperAndResolver(l_ProjectOwner);
      } else
      {
        controllerContract.registerWithConfig{value: _rent}(_name,address(this),_dur,_secret,address(resolverContract),address(this)); // ENS V2
      }

      resolverContract.setName(_dHash,string(abi.encodePacked(_name,tld())));
    }

    function _setOwner(bytes32 _domHash,address _owner,bytes32 _label,bytes32 _dHash) internal {

      resolverContract.setAddr(_domHash,_owner);                                                                       // e.g. _domHash = hash "vitalik.ethereum.eth"

      if (_label==0) {
        resolverContract.setText(_dHash,string('avatar'),string('https://www.ungravel.com/ung.png'));                  // set default group avatar UNG logo k_defaultAvatarURL
      }

      if (!isENSV3()) {
        base.ens().setOwner(_domHash,_owner);
      }
      else 
      {
        if (_label>0) {
          I_NWrapper(controllerContract.nameWrapper()).setSubnodeOwner(_dHash,labelStrFromBytes32(_label),_owner,0,0); // member label ownership
        }
        else
        {
          _allowNameWrapperAndResolver(_owner);
        }
      }
    }


    function register_ki_(bytes32[] calldata _m) external payable { 
                                                                                // _m[0] _domainHash
                                                                                // _m[1]  _secret
      uint256   _dur        = uint256(_m[2])>>128;                              // _m[2]  duration
      uint256   _rent       = uint256(_m[2])&k_rentMask;                        // _m[2]  rent
      string memory _name   = labelStrFromBytes32(_m[3]);                       // _m[3] domainName with length byte
      
      _register(_m[0],_rent,_name,_dur,_m[1]);
    }
    
    function _updateStructure(bytes32 _dHash,bytes calldata _data32) private {
      resolverContract.setABI(_dHash,32,abi.encodePacked(_data32));              // structure
      emit StructureDeployed(_dHash);
    }

    function update_P5q(bytes32 _dHash,bytes calldata data32) external payable {
      isInitiatorOrMember2(_dHash);
      _updateStructure(_dHash,data32);
    }

    function lock_dfs(bytes32 _dHash,bytes calldata data32) external payable {
      uint256 c = commitments[_dHash];      
      my_require((address(uint160(c & k_aMask))==msg.sender||msg.sender==address(this)) && installations[ uint64( (uint256(c & k_commitMask)>>160) & k_commit2Mask ) ]==0x0,"L"); // NOT locked getInstallTime(_dHash), getCommitment(_dHash) 
      
      uint64 hsh  = uint64( (uint256(c & k_commitMask)>>160) & k_commit2Mask );
      installations[hsh] = uint256( installations[hsh] & k_aMask ) + k_lockedMask;                      // saveInstallTime(1,_dHash)

      _updateStructure(_dHash,data32);
    }
    
    function registerAndLock_x3x(bytes32[] calldata _m, bytes calldata data32) external payable {
      bytes32 c = _m[0];                                                        // _m[0] = _domainHash
                                                                                // _m[1] = _secret
      // ---------------------------- register domain --------------------------

      uint256   _dur        = uint256(_m[2])>>128;                              // _m[2] = duration/rent
      uint256   _rent       = uint256(_m[2])&k_rentMask;                        // _m[2] = duration/rent
      string memory _name   = labelStrFromBytes32(_m[3]);                       // _m[3] = domainName with length byte  
      
      _register(c,_rent,_name,_dur,_m[1]);

      // ---------------------------- lock group structure----------------------
      
      my_require(installations[ uint64( (uint256(uint256(c) & k_commitMask)>>160) & k_commit2Mask ) ]==0x0,"r3"); // NOT locked getInstallTime(c), getCommitment(c) 
      
      uint64 hsh = uint64( (uint256(uint256(c) & k_commitMask)>>160) & k_commit2Mask );
      installations[hsh] = uint256( installations[hsh] & k_aMask ) + k_lockedMask;      // saveInstallTime(1,c)

      _updateStructure(c,data32);
    }
    
    function domainReport(string calldata _dom,uint command) external payable returns (uint256 report, address gwpc, address ptc, address gwfc, bytes memory structure) { 
      uint256 stamp     = 0;
      uint    nb        = 32;
      bytes memory abi32;
      
      bytes32 dHash      = getDomainHash(_dom);                                                                          // domain hash, ENS
      address owner      = base.ens().owner(dHash);                                                                      // ENS domain owner
      bool hasCommitment = uint64(getCommitment(dHash))>0x0;                                                             // hasCommitment
      
                                                                                report = uint256(getInstallTime(dHash) & 0x1);// locked group ? getInstallTime() =  domain install time (GWF)
      if (!base.ens().recordExists(dHash))                                      report = uint256(uint(report)+2);        // domain available - NOT existing
      if (owner == address(getGWProxy(dHash)) || owner == address(this))        report = uint256(uint(report)+4);        // domain contracted, GroupWalletProxy or this GWF contract is OWNER of domain
      if (base.ens().resolver(dHash) == address(resolverContract))              report = uint256(uint(report)+8);        // resolverContract is resolving domain, valid resolver
      if (hasCommitment)                                                        report = uint256(uint(report) + 16);     // domain with commitment
      if (resolverContract.addr(dHash) == address(this))                        report = uint256(uint(report) + 64);     // domain ENS resolves to the GWF contract, this contract

      if ((!isENSV3()&&(uint256(stringMemoryTobytes32(resolverContract.text(dHash,"use_color_table")))!=0x0000000f7573655f636f6c6f725f7461626c6500000000000000000000000000)) || (isENSV3()&&strlen(resolverContract.text(dHash,"use_color_table"))>0))
                                                                                report = uint256(uint(report)+2048);     // has colorTable
      if (getProxyToken(dHash) != k_add00)                                      report = uint256(uint(report)+4096);     // has proxyToken contract
      if (owner == k_add00)                                                     report = uint256(uint(report)+256);      // domain NOT owned owner = 0x000000000000000000000000000
      if (controllerContract.available(_dom))                                   report = uint256(uint(report)+512);      // domain is available
      if (owner == address(tx.origin))                                          report = uint256(uint(report)+1024);     // domain owned by default account

      if (isENSV3() && (owner == controllerContract.nameWrapper()))             report = uint256(uint(report)+8196);     // 8192+4 owner is NameWrapper, belonging to ENS V3, domain contracted
      
      if (hasCommitment) {                                                                                               // hasCommitment
        (nb, abi32) = resolverContract.ABI(dHash,128);                                                                   // isABI128
        if ((nb==128)&&(abi32.length>=224)&&((abi32.length%32)==0))             report = uint256(uint(report)+128);

        (nb, abi32) = resolverContract.ABI(dHash,32);                                                                    // isABIstructure, ABI32
        if ((nb==32)&&(abi32.length>32)&&(abi32.length<0x1000))                 report = uint256(uint(report)+32);
        
        nb = getOwners_internal( commitments[ dHash] ).length;                                                           // nb of members in group

        stamp = uint256(stringMemoryTobytes32(resolverContract.text(dHash,"use_timeStamp")));
        if (!isENSV3()&&(stamp==0x0000000d7573655f74696d655374616d70000000000000000000000000000000)) stamp = 0;          // timeStamp
      }
      
      report = uint256(stamp) + uint256(uint256(report)<<128) + uint256(nb << 64) + uint256(getInstallTime(dHash));
    
      if (command == 0) return (report,   getGWProxy(dHash),getProxyToken(dHash),address(this),abi32);                   // complete GWF domain report
      if (command == 1) return (stamp,    getGWProxy(dHash),getProxyToken(dHash),address(this),abi32);                   // only timeStamp of installation
    }
    
    function inviteInstallToken_q31n(bytes32[] memory _mem) public payable {      
      bytes32 _dHash = _mem[0];                                                 // _mem[0] domain hash identifying project/group
      uint l         = _mem.length-5;                                           // 5 words
      uint64 time    = uint64(block.timestamp*1000)&uint64(0xffffffffffff0000); // time % 0x1000
      uint256 amount = uint256(msg.value / uint256((l/5) + 3));                 // ether transferred to each member, 1 for PT, 1 for PGW, 1 remains for GWF
      uint256 c      = commitments[_dHash];
      
      {
       my_require((address(uint160(c & k_aMask))==msg.sender||address(msg.sender)==address(this))&&l>=10&&l<=160&&address(uint160(uint256(_mem[1])))!=k_add00&&msg.value>0,"w"); // msg.sender, l = l*5, masterCopy, value
      }
    
      address[] memory GWowners = new address[](l/5);                           // create array of owners()
      uint256[] memory GTowners = new uint256[]((l/5)+2);                       // create array of tokenOwner
      bytes memory abiCmd;                                                      // resolverContract.setABI
      
      {
        address o;
        bytes32 d;
        
        uint i=5;
        uint nb = 0;
        do {
          o = address(uint160(uint256(_mem[i+2]) & k_aMask));                   // owner address 
          d = _mem[i+1];                                                        // domainLabel hash 6, 11, 16...  _mem[i+1] = domain label hash 'vitalik.ethereum.eth'
          
          my_require(o!=k_add00 && _mem[i]!=0x0 && d!=0x0, "v4");               // 5, 9, 13... _mem[i] = label hash 'vitalik' || ENSV3: label str with length (bytes32)
          
          GWowners[nb] = o;                                                     // array of nb owners()
          GTowners[nb] = uint256(_mem[i+2]);                                    // _mem[i+2] array of tokenOwner uint256 contains share and address
          
          abiCmd       = abi.encodePacked(abiCmd,_mem[i+3]);                    // _mem[i+3] + 1 abi extra word 8, 13, 18... 

          my_require(payable(address(uint160(o))).send(amount),"v5");
          emit Deposit(address(uint160(o)),amount);
          
          if (!isENSV3()) {
            base.ens().setSubnodeRecord(_dHash, _mem[i], address(this), address(resolverContract), time); // assign ENS name e.g. vitalik.ethereum.eth
          } else
          {
            _setSubnodeRecordV3(_dHash,_mem[i],address(this),0);
          }

          {
            string memory avatar = bytesToStr(_mem[i+4],32);                    // _mem[i+4]  avatar url
            resolverContract.setText(d,string('avatar'),avatar);
          }

         _setOwner(d, o,_mem[i],_dHash);                                        // _mem[i+0] labelHash || labelStr

          nb++;
          i = i+5;
        } while ((i-5)<l&&i<=160);
      }
      
      {
        abiCmd = abi.encodePacked(k_abi80,k_abi80,k_abi80,bytes32(uint256((l/5)+1)<<5),abiCmd,_mem[2]);
        resolverContract.setABI(_dHash,128,abiCmd);                                                       // member addresses to ABI, one extra ABI 128 word      
      }

      ProxyGroupWallet proxyGW = new ProxyGroupWallet(address(uint160(uint256(_mem[4]))),concatString(_mem[3],tld()),reverseContract); // _mem[4] = masterCopy === GroupWalletMaster GWM, domainname

      {
        AbstractGWF_GWP(address(proxyGW)).newProxyGroupWallet_j5O{value: amount}( GWowners );             // initialize list of owners
        
        _setOwner(_dHash,address(proxyGW),0x0,_dHash);

        commitments[_dHash] = uint256(uint160(address(proxyGW)) & k_aMask) + uint256(c&k_commitMask);     // save initiator = GWP-GroupWalletProxy owner
        emit ProxyGroupWalletCreation(proxyGW);
      }

      {
        ProxyToken proxy = new ProxyToken( address(uint160(uint256(_mem[1]))) );                          // install ProxyToken contract and call the Token contract immediately, masterCopy
        
        GTowners[(l/5)+0] = uint256(_mem[3]);                                                             // tokenName = domain name
        GTowners[(l/5)+1] = uint256(uint160(address(proxyGW)));                                           // GroupWalletProxy contract address
        
        Abstract_TokenProxy(address(proxy)).newToken{value: amount}( GTowners );
      
        installations[ uint64( (uint256(c & k_commitMask)>>160) & k_commit2Mask ) ] = uint256( uint160(address(proxy)) ) + uint256( (uint256(time+1)<<160) & k_commitMask ); // saveProxyToken(address(proxy),_dHash) && saveInstallTime(time+1,_dHash)  
        emit ProxyTokenCreation(proxy);     
      }
    }

    function _setSubnodeRecordV3(bytes32 _dHash,bytes32 _labelStr,address _owner,uint64 _timeStamp) private {
      I_NWrapper(controllerContract.nameWrapper()).setSubnodeRecord(_dHash,labelStrFromBytes32(_labelStr),_owner,address(resolverContract),_timeStamp,0,0);
    }

   
    function upgradeGWF(bytes32 _dHash, address _GWF) external calledByUngravelGWP() onlyGWF(_GWF) payable { // *** MUST be called by a GWP contract belonging to UNGRAVEL ***
      uint256 c = commitments[_dHash];
      my_require(address( uint160(c & k_aMask) )==msg.sender&&version()<Abstract_GWF(_GWF).version(),"u");
      Abstract_GWF(_GWF).importGWP(_dHash,c,installations[ getCommitment(_dHash) ]);
    }
    
    function importGWP(bytes32 _dHash, uint256 commitment, uint256 installation) external onlyGWF(msg.sender) calledByUNG_GWP(_dHash) payable {
      my_require(version()>Abstract_GWF(msg.sender).version(),"i");
      
      address _GWP = Abstract_GWF(msg.sender).getGWProxy(_dHash);
      address _PT  = Abstract_GWF(msg.sender).getProxyToken(_dHash);

      my_require(installations[ uint64( (uint256(commitment & k_commitMask)>>160) & k_commit2Mask ) ]==0&&address(uint160(commitment & k_aMask))==_GWP&&_PT==address(uint160(uint256(installation) & k_aMask))&&commitments[_dHash]==0,"i5");

      installations[ uint64( (uint256(commitment & k_commitMask)>>160) & k_commit2Mask ) ] = installation;
      commitments[_dHash] = uint256(uint160(_GWP) & k_aMask) + uint256(commitment&k_commitMask);
    }
  
    function replicate_group_l9Y(bytes32[] calldata _m, bytes calldata data32, bytes32[] calldata _mem) external payable {
      uint256 v = 0;
    
      if (_m.length==4) {                                                       // replicate group
        v = uint256(_m[2])&k_rentMask;
        my_require(msg.value>0&&v>0&&msg.value>v&&address(uint160( commitments[ _m[0] ] & k_aMask ))==msg.sender,"G");
        (this).registerAndLock_x3x{value: v}( _m, data32 );
      }
      
      if (_m.length==1) {                                                       // confirm spin-off group
        my_require(msg.value>0,"H");
        isInitiatorOrMember2(_mem[0]);
        (this).lock_dfs(_mem[0], data32);
      }
      
      v = msg.value/25;                                                         // 4% contract provision funding ungravel
      
      (this).inviteInstallToken_q31n{value: uint256(msg.value)-v}( _mem );
    }
    
    function isInitiatorOrMember2(bytes32 _dHash) private view {                // update(), replicate()
      uint256 c = commitments[_dHash];
      if (address(uint160(c & k_aMask))==msg.sender) return;
      
      address[] memory memArr = getOwners_internal(c);                          // might be optimized in GW2
      uint l = memArr.length;

      uint i=0;
      
      do {
        if (memArr[i] == msg.sender) return;
        i++;
      } while(i<l);
      
      my_require(false, "owner2");
    }

    function isInitiatorOrMember(bytes32 _dHash) private view returns (address tProxy) {  // setTokenPrices(), transferOwner(), freezeToken(), transferToken(), transferTokenFrom()
      
      uint256 c = commitments[_dHash];
      if (address(uint160(c & k_aMask))==msg.sender) return address( uint160( uint256( installations[  uint64( (uint256( c & k_commitMask )>>160) & k_commit2Mask ) ] ) & k_aMask ) );

      address[] memory memArr = getOwners_internal(c);                          // might be optimized in GW2
      uint    l = memArr.length;
  
      uint index = 32;
      uint i=0;
      
      do {
        if (memArr[i] == msg.sender) index = i;
        i++;
      } while(i<l&&index==32);
      
      my_require(index>=0 && index<32, " - illegal/unknown initiator or owner.");
      
      return address( uint160( uint256( installations[  uint64( (uint256( c & k_commitMask )>>160) & k_commit2Mask ) ] ) & k_aMask ) );
    }
    
    function transferOwner_v3m(bytes32 _dHash, bytes memory data) public payable { 
      address tProxy = isInitiatorOrMember(_dHash);
      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
      emit TransferOwner(_dHash); 
    }

    function setTokenPrices_dgw(bytes32 _dHash, bytes memory data) public payable { 
      address tProxy = isInitiatorOrMember(_dHash);
      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
      emit SetPrices(_dHash);
    }
  
    function freezeToken_LGS(bytes32 _dHash, bytes memory data) public payable { 
      address tProxy = isInitiatorOrMember(_dHash);
      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
      emit FreezeToken(_dHash);
    }
    
    function TransferToken_8uf(bytes32 _dHash, bytes memory data) public payable { 
      address tProxy = isInitiatorOrMember(_dHash);
      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
      emit Transfer(address(this), address( uint160(uint256( uint256(mb32(substring(data,4,32))) & k_aMask ))), uint256( mb32(substring(data,36,32)) ) / 100);
    }

    function TransferTokenFrom_VCv(bytes32 _dHash, bytes memory data) public payable { 
      address tProxy = isInitiatorOrMember(_dHash);
      // solium-disable-next-line security/no-inline-assembly
      assembly {
        if eq(call(gas(), tProxy, 0, add(data, 0x20), mload(data), 0, 0), 0) { revert(0, 0) }
      }
      emit Transfer(address( uint160(uint256( uint256(mb32(substring(data,4,32))) & k_aMask ))), address( uint160(uint256( uint256(mb32(substring(data,36,32))) & k_aMask ))), uint256( mb32(substring(data,68,32)) ) / 100);
    }
    
    function withdraw() onlyByOwner() external {
      my_require(payable(address(uint160(msg.sender))).send(address(this).balance-1),"W");
    }

    function setAuctionsContract(Abs_AuctionRegistrar auctCtr) external onlyByOwner() payable {
      auctionContract = auctCtr;
    }
    
    function tld() public view returns (string memory) {
      uint chainId = block.chainid;
      if (chainId==1)        return ".eth";
      if (chainId==10)       return ".op";
      if (chainId==56)       return ".bsc";
      if (chainId==100)      return ".gnosis";
      if (chainId==130)      return ".uni";
      if (chainId==137)      return ".matic";
      if (chainId==1135)     return ".lisk";
      if (chainId==8453)     return ".base";
      if (chainId==42161)    return ".one";
      if (chainId==81457)    return ".blast";
      if (chainId==167000)   return ".tko";
      if (chainId==421614)   return ".arb";
      if (chainId==534352)   return ".scroll";
      if (chainId==11155111) return ".sepeth";
      if (chainId==11155420) return ".opt";
      return "";
    }
    
    function minimumPrePaid() private view returns (uint) {
      uint chainId = block.chainid;

      if (chainId==1)        return   0.01 ether;                               // .eth mainnet
      if (chainId==10)       return 0.0001 ether;                               // .op optmain
      if (chainId==56)       return   0.01 ether;                               // .bsc bscmain
      if (chainId==100)      return    0.1 ether;                               // .gnosis
      if (chainId==130)      return 0.0001 ether;                               // .uni
      if (chainId==137)      return    1.0 ether;                               // .matic
      if (chainId==1135)     return 0.0001 ether;                               // .lisk
      if (chainId==8453)     return 0.0001 ether;                               // .base base
      if (chainId==42161)    return 0.0001 ether;                               // .one arbitrum one = arbmain
      if (chainId==81457)    return 0.0001 ether;                               // .blast
      if (chainId==167000)   return   0.01 ether;                               // .tko taiko
      if (chainId==421614)   return    0.1 ether;                               // .arb sepolia testnet
      if (chainId==534352)   return  0.001 ether;                               // .scroll
      if (chainId==11155111) return   0.25 ether;                               // .sepeth
      if (chainId==11155420) return    0.1 ether;                               // .opt
      return 0.15 ether;                                                        // .eth ganache & other chains
    }

    function version() public pure returns(uint256 v) {
      return 20010088;
    }
    
    function receivePayment() external payable returns(bool) {                  // receive e.g. ungravel license fees
      emit Deposit(msg.sender, msg.value);
      return true;
    }

    fallback() external payable
    {
      my_require(false, "fb");
    }
    
    receive() external payable {      
      if (msg.value==0) return;                                                 // no payment at all
      emit Deposit(msg.sender, msg.value);
    }

    constructor (AbstractETHRegController _controller,AbstractGWF_Resolver _resolver,AbstractBaseRegistrar _base,AbstractGWF_ENS _ens,AbstractGWF_ReverseRegistrar _reverse, NWrapper _nwrapper,Abs_AuctionRegistrar _auctionMaster) payable {      
      GWFowner                          = tx.origin;
      
      controllerContract                = _controller;
      resolverContract                  = _resolver;
      base                              = _base;
      ens                               = _ens;
      reverseContract                   = _reverse;
      ensNameWrapper                    = _nwrapper;
      auctionContract                   = _auctionMaster;
    }
}