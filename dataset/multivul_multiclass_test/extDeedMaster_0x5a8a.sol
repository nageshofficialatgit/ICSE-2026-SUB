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


// ******************************* EXT DEED MASTER CONTRACT ********************
// An external bidder is unknown, anonymous and not a member of the group. A proxy bidding contract is deployed
// for each external bidder beeing invited to participate in a Funding Auction.

abstract contract ABS_Reg {
  function state_pln(bytes32 _hash) public view virtual returns (uint);
  function saveExtDeedCntr_gm9(address _sender,bytes32 _hash,uint _value) public payable virtual;
  function unsealExternalBid_qfG(bytes32 _hash) public payable virtual;
  function finalizeExternalAuction_WmS(bytes32 _hash) public payable virtual;
  function cancelExternalBid_9ig(bytes32 seal, bytes32 hash) public payable virtual;
}

bytes32 constant kkk = 0x4db45745d63e3d3fca02d388bb6d96a256b72fa6a5ca7e7b2c10c90c84130f3b;


contract extDeedMaster {
  address internal masterCopy;
  address public  owner;
  uint64  public  creationDate;
  ABS_Reg public  registrar;
  bytes32 public  lhash;

  event DeedCreated(address indexed,bytes32 indexed);
  event NewBid(address indexed);
  event RevealBid(address indexed);
  event CancelBid(address indexed);

  constructor(address _masterCopy) payable
  {
    masterCopy   = _masterCopy;
    owner        = tx.origin;
    registrar    = ABS_Reg(msg.sender);
    creationDate = uint64(block.timestamp);
  }

  function getMasterCopy() public view returns (address) {
    return masterCopy;
  }

  function adjustBal_1k3(uint newValue) public payable {                        // 0x0000f6a6
    if (address(this).balance<=newValue) return;
    require(msg.sender==address(registrar)&&address(this).balance>0&&payable(address(uint160(owner))).send(address(this).balance-newValue),"A");
  }

  function closeDeed_igk(address receiver) public payable {                     // 0x00004955
    require(msg.sender==address(registrar)&&payable(address( (receiver!=address(0x0)) ? receiver : owner )).send(address(this).balance),"B");
  }
  
  receive() external payable {                                                  // receiving fallback function, catches all extDeedProxy calls
    uint lstate = registrar.state_pln(lhash);
    
    if (lstate==1) {                                                            // OPEN for bidding
      require(lhash!=0x0&&msg.value>0&&address(this).balance==msg.value,"C");
      owner = msg.sender;
      registrar.saveExtDeedCntr_gm9(msg.sender,lhash,msg.value);
      emit NewBid(msg.sender);
    } else
    {

      require(lhash!=0x0&&msg.value==0&&owner==msg.sender,"D");                 // only Deed owner calls without ETH

      if (lstate==4) {                                                      
        registrar.unsealExternalBid_qfG(lhash);                                 // REVEAL phase
        emit RevealBid(msg.sender);
      } else
      {
        if (lstate==2) {                                                        // FINALIZE phase
          registrar.finalizeExternalAuction_WmS(lhash);
        } else
        {
          if (lstate==6) {                                                      // CANCEL - auction done, no time-out
            registrar.cancelExternalBid_9ig(keccak256(abi.encode(lhash,owner,address(this).balance,kkk)),lhash);
            emit CancelBid(msg.sender);
          } else
          {
            if (lstate==0) {                                                    // TIMEOUT - auction done, no bidding revealed and finalized
              require(payable(owner).send(address(this).balance),"E");
              emit CancelBid(msg.sender);
            } else
            {                                                                   // unknown state --> throw an error and revert
              require(false,"F");                                               // fallback - unknown auction state
            }
          }
        }
      }
    }
  }
}