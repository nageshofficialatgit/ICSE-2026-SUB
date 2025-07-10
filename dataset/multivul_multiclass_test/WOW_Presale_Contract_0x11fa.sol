// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

interface AggregatorInterface {
    function latestAnswer() external view returns (int256);
}

library SafeMath {
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }
    function tryMul(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }
}

interface IBEP20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

contract WOW_Presale_Contract is Ownable {
    using SafeMath for uint256;
    
    IBEP20 public immutable baseToken;
    IBEP20 public immutable presaleToken;
    uint256 public presaleRate = 100000000000000;
    AggregatorInterface public priceFeed;

    event TokensSwapped(address indexed recipient, uint256 baseAmount, uint256 presaleAmount);
    event BaseTokenWithdrawn(address indexed recipient, uint256 amount);
    event PresaleTokenWithdrawn(address indexed recipient, uint256 amount);

    constructor(address _baseToken, address _presaleToken, address _priceFeed) {
        baseToken = IBEP20(_baseToken);
        presaleToken = IBEP20(_presaleToken);
        priceFeed = AggregatorInterface(_priceFeed);
    }

    function getReceivablePresaleTokens(uint256 amount, bool isETH) external view returns (uint256) {
        require(amount > 0, "WOW Presale: amount must be greater than zero");
        if (isETH) {
            int256 ethPrice = priceFeed.latestAnswer();
            require(ethPrice > 0, "WOW Presale: Invalid ETH price from oracle");
            uint256 usdtEquivalent = (amount * uint256(ethPrice)) / (10**8) / (10**12);
            return usdtEquivalent.mul(presaleRate);
        } else {
            return amount.mul(presaleRate);
        }
    }
    function swapForUSDT(uint256 amount) external {
        require(amount > 0, "WOW Presale: amount must be greater than zero");
        require(baseToken.transferFrom(_msgSender(), address(this), amount), "WOW Presale: USDT transfer failed");
        uint256 presaleAmount = amount.mul(presaleRate);
        require(presaleToken.transfer(_msgSender(), presaleAmount), "WOW Presale: Presale token transfer failed");
        emit TokensSwapped(_msgSender(), amount, presaleAmount);
    }
    function swapForETH() external payable {
        require(msg.value > 0, "WOW Presale: ETH value must be greater than zero");
        int256 ethPrice = priceFeed.latestAnswer();
        require(ethPrice > 0, "WOW Presale: Invalid ETH price from oracle");
        uint256 usdtEquivalent = (msg.value * uint256(ethPrice)) / (10**8) / (10**12);
        uint256 presaleAmount = usdtEquivalent.mul(presaleRate);
        require(presaleToken.transfer(_msgSender(), presaleAmount), "WOW Presale: Presale token transfer failed");
        emit TokensSwapped(_msgSender(), usdtEquivalent, presaleAmount);
    }
    function updatePresaleRate(uint256 newRate) external onlyOwner {
        require(newRate > 0, "WOW Presale: rate must be greater than zero");
        presaleRate = newRate;
    }
    function withdrawBaseToken(address recipient, uint256 amount) external onlyOwner {
        require(baseToken.transfer(recipient, amount), "WOW Presale: base token transfer failed");
        emit BaseTokenWithdrawn(recipient, amount);
    }
    function withdrawPresaleToken(address recipient, uint256 amount) external onlyOwner {
        require(presaleToken.transfer(recipient, amount), "WOW Presale: presale token transfer failed");
        emit PresaleTokenWithdrawn(recipient, amount);
    }
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "WOW Presale: No ETH balance to withdraw");
        payable(_msgSender()).transfer(balance);
    }
    function getLatestPrice() external view returns (int256) {
        return priceFeed.latestAnswer();
    }
}