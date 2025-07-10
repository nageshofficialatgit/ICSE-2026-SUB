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

/// @title Proxy - Generic proxy contract allows to execute all transactions applying the code of a master contract.
/// @author Stefan George - <stefan@gnosis.pm> /// ProxyGroupWallet adapted and applied for GroupWallet by pepihasenfuss.eth

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
      _reverse.setName(_domain); // if (block.chainid!=1) 
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