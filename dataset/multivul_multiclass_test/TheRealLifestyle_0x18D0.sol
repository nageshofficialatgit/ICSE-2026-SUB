pragma solidity 0.8.25;

// SPDX-License-Identifier: MIT


abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }
}

interface IERC20 {
    /**
     * @dev Returns the amount of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `recipient`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `sender` to `recipient` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IERC20Metadata is IERC20{
    /**
     * @dev Returns the name of the token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the symbol of the token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the decimals places of the token.
     */
    function decimals() external view returns (uint8);
}

contract ERC20 is Context, IERC20, IERC20Metadata {
    mapping(address => uint256) private _balances;

    mapping(address => mapping(address => uint256)) private _allowances;

    uint256 private _totalSupply;

    string private _name;
    string private _symbol;

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual override returns (string memory) {
        return _name;
    }

    function contractEthBalance() external view returns (uint256) {
        return address(this).balance;
    }


    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals used to get its user representation.
     * For example, if `decimals` equals `2`, a balance of `505` tokens should
     * be displayed to a user as `5.05` (`505 / 10 ** 2`).
     *
     * Tokens usually opt for a value of 18, imitating the relationship between
     * Ether and Wei. This is the default value returned by this function, unless
     * it's overridden.
     *
     * NOTE: This information is only used for _display_ purposes: it in
     * no way affects any of the arithmetic of the contract, including
     * {IERC20-balanceOf} and {IERC20-transfer}.
     */
    function decimals() public view virtual override returns (uint8) {
        return 9;
    }

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev See {IERC20-transfer}.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - the caller must have a balance of at least `amount`.
     */
    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, amount);
        return true;
    }

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-approve}.
     *
     * NOTE: If `amount` is the maximum `uint256`, the allowance is not updated on
     * `transferFrom`. This is semantically equivalent to an infinite approval.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }

    /**
     * @dev See {IERC20-transferFrom}.
     *
     * Emits an {Approval} event indicating the updated allowance. This is not
     * required by the EIP. See the note at the beginning of {ERC20}.
     *
     * NOTE: Does not update the allowance if the current allowance
     * is the maximum `uint256`.
     *
     * Requirements:
     *
     * - `from` and `to` cannot be the zero address.
     * - `from` must have a balance of at least `amount`.
     * - the caller must have allowance for ``from``'s tokens of at least
     * `amount`.
     */
    function transferFrom(address from, address to, uint256 amount) public virtual override returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }

    /**
     * @dev Atomically increases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function increaseAllowance(address spender, uint256 addedValue) public virtual returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, allowance(owner, spender) + addedValue);
        return true;
    }

    /**
     * @dev Atomically decreases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     * - `spender` must have allowance for the caller of at least
     * `subtractedValue`.
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public virtual returns (bool) {
        address owner = _msgSender();
        uint256 currentAllowance = allowance(owner, spender);
        require(currentAllowance >= subtractedValue, "TRLX: decreased allowance below zero");
        unchecked {
            _approve(owner, spender, currentAllowance - subtractedValue);
        }

        return true;
    }

    /**
     * @dev Moves `amount` of tokens from `from` to `to`.
     *
     * This internal function is equivalent to {transfer}, and can be used to
     * e.g. implement automatic token fees, slashing mechanisms, etc.
     *
     * Emits a {Transfer} event.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `from` must have a balance of at least `amount`.
     */
    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "TRLX: transfer from the zero address");
        require(to != address(0), "TRLX: transfer to the zero address");

        _beforeTokenTransfer(from, to, amount);

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "TRLX: transfer amount exceeds balance");
        unchecked {
            _balances[from] = fromBalance - amount;
            // Overflow not possible: the sum of all balances is capped by totalSupply, and the sum is preserved by
            // decrementing then incrementing.
            _balances[to] += amount;
        }

        emit Transfer(from, to, amount);

        _afterTokenTransfer(from, to, amount);
    }

    /** @dev Creates `amount` tokens and assigns them to `account`, increasing
     * the total supply.
     *
     * Emits a {Transfer} event with `from` set to the zero address.
     *
     * Requirements:
     *
     * - `account` cannot be the zero address.
     */
    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "TRLX: mint to the zero address");

        _beforeTokenTransfer(address(0), account, amount);

        _totalSupply += amount;
        unchecked {
            // Overflow not possible: balance + amount is at most totalSupply + amount, which is checked above.
            _balances[account] += amount;
        }
        emit Transfer(address(0), account, amount);

        _afterTokenTransfer(address(0), account, amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`, reducing the
     * total supply.
     *
     * Emits a {Transfer} event with `to` set to the zero address.
     *
     * Requirements:
     *
     * - `account` cannot be the zero address.
     * - `account` must have at least `amount` tokens.
     */
    function _burn(address account, uint256 amount) internal virtual {
        require(account != address(0), "TRLX: burn from the zero address");

        _beforeTokenTransfer(account, address(0), amount);

        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount, "TRLX: burn amount exceeds balance");
        unchecked {
            _balances[account] = accountBalance - amount;
            // Overflow not possible: amount <= accountBalance <= totalSupply.
            _totalSupply -= amount;
        }

        emit Transfer(account, address(0), amount);

        _afterTokenTransfer(account, address(0), amount);
    }

    /**
     * @dev Sets `amount` as the allowance of `spender` over the `owner` s tokens.
     *
     * This internal function is equivalent to `approve`, and can be used to
     * e.g. set automatic allowances for certain subsystems, etc.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `owner` cannot be the zero address.
     * - `spender` cannot be the zero address.
     */
    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "TRLX: approve from the zero address");
        require(spender != address(0), "TRLX: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `amount`.
     *
     * Does not update the allowance amount in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Might emit an {Approval} event.
     */
    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "TRLX: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }

    /**
     * @dev Hook that is called before any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
     * will be transferred to `to`.
     * - when `from` is zero, `amount` tokens will be minted for `to`.
     * - when `to` is zero, `amount` of ``from``'s tokens will be burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual {}

    /**
     * @dev Hook that is called after any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
     * has been transferred to `to`.
     * - when `from` is zero, `amount` tokens have been minted for `to`.
     * - when `to` is zero, `amount` of ``from``'s tokens have been burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _afterTokenTransfer(address from, address to, uint256 amount) internal virtual {}
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() external virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

interface IDexRouter {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function swapExactTokensForETHSupportingFeeOnTransferTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external;
}

interface IDexFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract TheRealLifestyle is ERC20, Ownable {

    mapping(address => uint256) private _holderLastTransferBlock; // to hold last Transfers temporarily during launch
    bool public transferDelayInEffect = true;
    bool public limitsInEffect = true;

    bool private swapping;
    uint256 public swapTokensAtAmt;    

    uint256 public constant FEE_DIVISOR = 10000;

    mapping (address => bool) public exemptFromTaxes;
    mapping (address => bool) public exemptFromLimits;

    bool public tradingActive;

    mapping (address => bool) public isLPPair;

    address public opsAddress;

    uint256 public maxTrans;
    uint256 public maxWallet;

    uint256 public buyTax;
    uint256 public sellTax;

    address public immutable lpPair;
    IDexRouter public immutable dexRouter;

    

    // events

    event UpdatedMaxTrans(uint256 newMax);
    event UpdatedWalletLimit(uint256 newMax);
    event SetExemptFromFees(address _address, bool _isExempt);
    event SetExemptFromLimits(address _address, bool _isExempt);
    event RemovedLimits();
    event UpdatedBuyTax(uint256 newAmt);
    event UpdatedSellTax(uint256 newAmt);

    // constructor

    constructor()
        ERC20("The Real Lifestyle", "TRLX")
    {   
        address newOwner = _msgSender();
        _mint(newOwner, 1_000_000_000 * 1e9);
        uint256 _totalSupply = totalSupply();

        address _v2Router;

        // @dev assumes WETH pair
        if(block.chainid == 1){
            _v2Router = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
        } else if(block.chainid == 11155111){
            _v2Router = 0xC532a74256D3Db42D0Bf7a0400fEFDbad7694008;
        } else {
            revert("Chain not configured");
        }

        dexRouter = IDexRouter(_v2Router);

        maxTrans = _totalSupply * 20 / 1000;
        maxWallet = _totalSupply * 20 / 1000;
        swapTokensAtAmt = _totalSupply * 25 / 100000;

        opsAddress = 0x66457dbdee87FC2D45208BD8C0bE7e150878099f;

        buyTax = 1500; // 1% = 100
        sellTax = 1500; // 1% = 100

        lpPair = IDexFactory(dexRouter.factory()).createPair(address(this), dexRouter.WETH());

        isLPPair[lpPair] = true;

        exemptFromLimits[lpPair] = true;
        exemptFromLimits[newOwner] = true;
        exemptFromLimits[address(this)] = true;
        exemptFromLimits[address(dexRouter)] = true;
        

        exemptFromTaxes[newOwner] = true;
        exemptFromTaxes[address(this)] = true;
        exemptFromTaxes[address(dexRouter)] = true;
        
 
        _approve(address(this), address(dexRouter), type(uint256).max);
        transferOwnership(newOwner);
    }

    // owner functions

    function changeExemptFromFees(address _address, bool _isExempt) external onlyOwner {
        require(_address != address(0), "Zero Address");
        exemptFromTaxes[_address] = _isExempt;
        emit SetExemptFromFees(_address, _isExempt);
    }

    function changeExemptFromLimits(address _address, bool _isExempt) external onlyOwner {
        require(_address != address(0), "Zero Address");
        if(!_isExempt){
            require(_address != lpPair, "Pair");
        }
        exemptFromLimits[_address] = _isExempt;
        emit SetExemptFromLimits(_address, _isExempt);
    }

    function setMaxTransaction(uint256 newNumInTokens) external onlyOwner {
        require(newNumInTokens >= (totalSupply() * 5 / 1000)/(10**decimals()), "Must be >= 0.5%");
        maxTrans = newNumInTokens * (10**decimals());
        emit UpdatedMaxTrans(maxTrans);
    }

    function setMaxWallet(uint256 newNumInTokens) external onlyOwner {
        require(newNumInTokens >= (totalSupply() * 1 / 100)/(10**decimals()), "Must be >= 1%");
        maxWallet = newNumInTokens * (10**decimals());
        emit UpdatedWalletLimit(maxWallet);
    }

    function setTaxes(uint256 _buyTax, uint256 _sellTax) external onlyOwner {
        buyTax = _buyTax;
        emit UpdatedBuyTax(buyTax);
        sellTax = _sellTax;
        emit UpdatedSellTax(sellTax);
    }

    function enableTRLXTrade() external onlyOwner {
        require(!tradingActive, "Trading active");
        tradingActive = true;
    }

    function removeAllRestrictions() external onlyOwner {
        limitsInEffect = false;
        transferDelayInEffect = false;
        maxTrans = totalSupply();
        maxWallet = totalSupply();
        emit RemovedLimits();
    }

    function disableTransferDelay() external onlyOwner {
        transferDelayInEffect = false;
    }

    function setOpsAddress(address _address) external onlyOwner {
        require(_address != address(0), "zero address");
        opsAddress = _address;
    }

    receive() external payable {}

    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual override {
        
        if(exemptFromTaxes[from] || exemptFromTaxes[to] || swapping){
            super._transfer(from,to,amount);
            return;
        }

        require(tradingActive, "Trading not active");

        amount -= routeTax(from, to, amount);

        if(limitsInEffect){
            checkRestrictions(from, to, amount);
        }

        super._transfer(from,to,amount);
    }

    function checkRestrictions(address from, address to, uint256 amount) internal {
        if (transferDelayInEffect){
            if (to != address(dexRouter) && !isLPPair[to]){
                require(_holderLastTransferBlock[tx.origin] < block.number, "Transfer Delay enabled.");
                _holderLastTransferBlock[tx.origin] = block.number;
            }
        }

        // buy
        if (isLPPair[from] && !exemptFromLimits[to]) {
            require(amount <= maxTrans, "Max tx exceeded.");
            require(amount + balanceOf(to) <= maxWallet, "Max wallet exceeded");
        } 
        // sell
        else if (isLPPair[to] && !exemptFromLimits[from]) {
            require(amount <= maxTrans, "Max tx exceeded.");
        }
        else if(!exemptFromLimits[to]) {
            require(amount + balanceOf(to) <= maxWallet, "Max wallet exceeded");
        }
    }

    function routeTax(address from, address to, uint256 amount) internal returns (uint256){
        if(balanceOf(address(this)) >= swapTokensAtAmt && !swapping && !isLPPair[from]) {
            swapping = true;
            swap();
            swapping = false;
        }
        
        uint256 tax = 0;

        // on sell
        if (isLPPair[to] && sellTax > 0){
            tax = amount * sellTax / FEE_DIVISOR;
        }
        // on buy
        else if(isLPPair[from] && buyTax > 0) {
            tax = amount * buyTax / FEE_DIVISOR;
        }
        
        if(tax > 0){    
            super._transfer(from, address(this), tax);
        }
        
        return tax;
    }

    function swapTokensForETH(uint256 tokenAmt) private {

        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = dexRouter.WETH();

        dexRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmt,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swap() private {

        uint256 contractBalance = balanceOf(address(this));
        
        if(contractBalance == 0) {return;}

        if(contractBalance > swapTokensAtAmt * 40){
            contractBalance = swapTokensAtAmt * 40;
        }
        
        swapTokensForETH(contractBalance);
            
        if(address(this).balance > 0){
            bool success;
            (success, ) = opsAddress.call{value: address(this).balance}("");
        }
    }
}