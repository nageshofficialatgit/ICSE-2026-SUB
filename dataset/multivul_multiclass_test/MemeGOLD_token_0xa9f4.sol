// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

//////////////////////////////  mGOLD  /////////////////////////////////////
//███╗   ███╗███████╗███╗   ███╗███████╗███████╗██╗    ██╗ █████╗ ██████╗ //
//████╗ ████║██╔════╝████╗ ████║██╔════╝██╔════╝██║    ██║██╔══██╗██╔══██╗//
//██╔████╔██║█████╗  ██╔████╔██║█████╗  ███████╗██║ █╗ ██║███████║██████╔╝//
//██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ╚════██║██║███╗██║██╔══██║██╔═══╝ //
//██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗███████║╚███╔███╔╝██║  ██║██║     //
//╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝     //
//////////////////////////////  .money  ////////////////////////////////////


abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

}

contract ERC20 is Context, IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        uint256 currentAllowance = _allowances[sender][_msgSender()];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        _approve(sender, _msgSender(), currentAllowance - amount);
        return true;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal virtual {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");

        uint256 senderBalance = _balances[sender];
        require(senderBalance >= amount, "ERC20: transfer amount exceeds balance");
        _balances[sender] = senderBalance - amount;
        _balances[recipient] += amount;

        emit Transfer(sender, recipient, amount);
    }

    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: mint to the zero address");

        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}

contract MemeGOLD_token is Ownable, ERC20 {
    mapping(address => bool) private _eco;        
    mapping(address => bool) private _trust;       
    mapping(address => bool) private _lqpool;     
    mapping(address => bool) private _uniswapPools; 

    address private _stakeAddress; 
    bool private _isFree = false; 

    event IEco(address indexed contractAddress);
    event OEco(address indexed contractAddress);
    event ITrust(address indexed account);
    event OTrust(address indexed account);
    event ILqPool(address indexed account);
    event OLqPool(address indexed account);
    event StakeAddressUpdated(address indexed newStakeAddress);
    event IUniswapPool(address indexed poolAddress);
    event OUniswapPool(address indexed poolAddress);


    constructor() ERC20("memeGOLD", "mGOLD") {
        _mint(msg.sender, 100_000 * 10 ** decimals()); // Total 100,000 mGOLD
    }



function iEco(address contractAddress) external onlyOwner {
    require(contractAddress != address(0), "Invalid contract address");
    require(!_eco[contractAddress], "listed");
    _eco[contractAddress] = true;
    emit IEco(contractAddress);
}

    function oEco(address contractAddress) external onlyOwner {
        require(_eco[contractAddress], "not in eco");
        _eco[contractAddress] = false;
        emit OEco(contractAddress);
    }

    function iTrust(address account) external onlyOwner {
        require(account != address(0), "Invalid address");
        _trust[account] = true;
        emit ITrust(account);
    }

    function oTrust(address account) external onlyOwner {
        require(_trust[account], "not in trust");
        _trust[account] = false;
        emit OTrust(account);
    }

    function iLqPool(address account) external onlyOwner {
        require(account != address(0), "Invalid address");
        _lqpool[account] = true;
        emit ILqPool(account);
    }

    function oLqPool(address account) external onlyOwner {
        require(_lqpool[account], "not in lqpool");
        _lqpool[account] = false;
        emit OLqPool(account);
    }

    function iUniswapPool(address poolAddress) external onlyOwner {
        require(poolAddress != address(0), "Invalid pool address");
        _uniswapPools[poolAddress] = true;
        emit IUniswapPool(poolAddress);
    }

    function oUniswapPool(address poolAddress) external onlyOwner {
        require(_uniswapPools[poolAddress], "not in uniswap");
        _uniswapPools[poolAddress] = false;
        emit OUniswapPool(poolAddress);
    }

    function setStakeAddress(address stakeAddress) external onlyOwner {
        require(stakeAddress != address(0), "Invalid stake address");
        _stakeAddress = stakeAddress;
        emit StakeAddressUpdated(stakeAddress);
    }

    function setFreeMode(bool isEnabled) external onlyOwner {
        _isFree = isEnabled;
    }    

    function _isShift(address from, address to) internal view returns (bool) {
        if (_isFree) {
            return true; 
        }
        if (from == owner() || _trust[from]) {
            return true; 
        }
        if (_lqpool[from]) {
            return to == _stakeAddress; 
        }
        if (_uniswapPools[from]) {
            return true; 
        }
        return _eco[to]; 
    }

    function _transfer(address sender, address recipient, uint256 amount) internal override {
        require(_isShift(sender, recipient), "error");
        super._transfer(sender, recipient, amount);
    }

    function recoverERC20(address tokenAddress, uint256 tokenAmount) external onlyOwner {
        require(tokenAddress != address(0), "Invalid token address");
        IERC20(tokenAddress).transfer(owner(), tokenAmount);
    }

    function recoverETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }

    receive() external payable {}
}